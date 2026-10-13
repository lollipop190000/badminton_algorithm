from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas import (  # noqa: E402
    CQLResult,
    ExampleDetail,
    ExampleSummary,
    GRUResult,
    RallyRequest,
    RallyResponse,
    RecommendedAction,
    StrokeInput,
    StrokeResult,
)
from src.cql_model import DiscreteCQLNetwork  # noqa: E402
from src.gru_model import ShuttleGRU  # noqa: E402
from src.train_gru import create_model  # noqa: E402


SHOT_LABELS_KO = {
    "defensive_shot": "수비형 샷",
    "drive": "드라이브",
    "drop": "드롭",
    "hairpin": "헤어핀",
    "high_clear": "하이클리어",
    "long_service": "롱서비스",
    "push": "푸시",
    "short_service": "숏서비스",
    "smash": "스매시",
    "under_clear": "언더클리어",
}


class RallySimulator:
    def __init__(self, root: Path = PROJECT_ROOT) -> None:
        self.root = root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gru_checkpoint = torch.load(
            root / "outputs/gru/model/best_gru.pt", map_location=self.device, weights_only=False
        )
        self.gru: ShuttleGRU = create_model(
            self.gru_checkpoint["config"], self.gru_checkpoint["preprocessing"]
        ).to(self.device)
        self.gru.load_state_dict(self.gru_checkpoint["model_state_dict"])
        self.gru.eval()

        self.cql_checkpoint = torch.load(
            root / "outputs/cql_v2/model/best_cql_model.pt", map_location=self.device, weights_only=False
        )
        self.actions = (
            pd.DataFrame(self.cql_checkpoint["actions"])
            .sort_values("action_id")
            .reset_index(drop=True)
        )
        self.actions["action_id"] = self.actions["action_id"].astype(int)
        self.actions["target_zone"] = self.actions["target_zone"].astype(int)
        self.cql = DiscreteCQLNetwork(
            len(self.cql_checkpoint["state_columns"]),
            len(self.actions),
            list(self.cql_checkpoint["config"]["hidden_layers"]),
        ).to(self.device)
        self.cql.load_state_dict(self.cql_checkpoint["model_state_dict"])
        self.cql.eval()
        self.state_report = self.cql_checkpoint["state_report"]
        self.action_by_label = {
            str(row.actual_action): int(row.action_id) for row in self.actions.itertuples()
        }
        support_path = root / "outputs/cql_v2/data/context_action_support.csv"
        support = pd.read_csv(support_path)
        self.context_counts = {
            (str(row.context_key), int(row.action_id)): int(row.context_train_count)
            for row in support.itertuples()
        }
        self.test_frame = pd.read_csv(root / "data/processed/gru/test_gru_features.csv")
        self._example_ids = self._compatible_example_ids()
        valid_ids = set(self._example_ids)
        self._example_frames = {
            str(rally_id): part.sort_values("stroke_no").reset_index(drop=True)
            for rally_id, part in self.test_frame.groupby("rally_id", sort=False)
            if str(rally_id) in valid_ids
        }
        self._example_summaries = [self._example_summary(rally_id) for rally_id in self._example_ids]

    @property
    def health(self) -> dict[str, Any]:
        return {
            "loaded": True,
            "device": str(self.device),
            "gru_version": f"best epoch {self.gru_checkpoint['best_epoch']}",
            "cql_version": "CQL v2",
            "cql_actions": len(self.actions),
            "examples": len(self._example_ids),
        }

    def _compatible_example_ids(self) -> list[str]:
        valid: list[str] = []
        ordered = self.test_frame.sort_values(["rally_id", "stroke_no"])
        for rally_id, part in ordered.groupby("rally_id", sort=False):
            shots = part["shot_type_standardized"].astype(str).tolist()
            hitters = part["hitter"].astype(str).tolist()
            zones = [int(float(value)) for value in part["shot_target_zone"]]
            if not shots or shots[0] not in {"short_service", "long_service"}:
                continue
            if any(shots[index] in {"short_service", "long_service"} for index in range(1, len(shots))):
                continue
            if any(hitters[index] == hitters[index - 1] for index in range(1, len(hitters))):
                continue
            if any(zone == 0 for zone in zones[:-1]):
                continue
            valid.append(str(rally_id))
        return valid

    def list_examples(self, query: str = "", limit: int = 500) -> list[ExampleSummary]:
        query_folded = query.strip().casefold()
        rows: list[ExampleSummary] = []
        for summary in self._example_summaries:
            haystack = " ".join(
                [summary.rally_id, summary.player_a_name, summary.player_b_name, summary.tournament, summary.round, summary.label]
            ).casefold()
            if query_folded and query_folded not in haystack:
                continue
            rows.append(summary)
            if len(rows) >= limit:
                break
        return rows

    def _example_summary(self, rally_id: str) -> ExampleSummary:
        part = self._example_frames[rally_id]
        first = part.iloc[0]
        player_a = str(first["player_A"])
        player_b = str(first["player_B"])
        tournament = str(first.get("tournament", ""))
        round_name = str(first.get("round_standardized", first.get("round", "")))
        return ExampleSummary(
            rally_id=rally_id,
            match_id=str(first["match_id"]),
            player_a_name=player_a,
            player_b_name=player_b,
            tournament=tournament,
            round=round_name,
            strokes=len(part),
            label=f"{player_a} vs {player_b} · {len(part)}타구 · {rally_id}",
        )

    def get_example(self, rally_id: str) -> ExampleDetail:
        if rally_id not in self._example_ids:
            raise KeyError(rally_id)
        part = self._example_frames[rally_id]
        first = part.iloc[0]
        strokes = [
            StrokeInput(
                hitter=str(row.hitter),
                shot_type=str(row.shot_type_standardized),
                target_zone=int(float(row.shot_target_zone)),
                player_a_position={"x_m": float(row.player_x), "y_m": float(row.player_y)},
                player_b_position={"x_m": float(row.opponent_x), "y_m": float(row.opponent_y)},
                delta_time_s=float(row.delta_time_s),
            )
            for row in part.itertuples()
        ]
        return ExampleDetail(
            rally_id=rally_id,
            request=RallyRequest(
                player_a_name=str(first["player_A"]),
                player_b_name=str(first["player_B"]),
                strokes=strokes,
            ),
        )

    @staticmethod
    def _raw_features(strokes: list[StrokeInput]) -> np.ndarray:
        rows: list[list[float]] = []
        previous_a: tuple[float, float] | None = None
        previous_b: tuple[float, float] | None = None
        for stroke in strokes:
            a = (stroke.player_a_position.x_m, stroke.player_a_position.y_m)
            b = (stroke.player_b_position.x_m, stroke.player_b_position.y_m)
            if previous_a is None:
                move_a = move_b = 0.0
            else:
                move_a = math.dist(a, previous_a)
                move_b = math.dist(b, previous_b)  # type: ignore[arg-type]
            if stroke.delta_time_s > 0:
                speed_a = move_a / stroke.delta_time_s
                speed_b = move_b / stroke.delta_time_s
            else:
                speed_a = speed_b = math.nan
            rows.append(
                [
                    a[0], a[1], b[0], b[1], stroke.delta_time_s,
                    move_a, speed_a, move_b, speed_b,
                ]
            )
            previous_a, previous_b = a, b
        return np.asarray(rows, dtype=np.float64)

    @staticmethod
    def _scale_columns(raw: np.ndarray, names: list[str], state: dict[str, Any]) -> np.ndarray:
        scaled = np.zeros_like(raw, dtype=np.float32)
        for index, name in enumerate(names):
            params = state[name]
            values = raw[:, index].copy()
            values[~np.isfinite(values)] = float(params["median"])
            values = np.clip(values, float(params["clip_lower"]), float(params["clip_upper"]))
            scaled[:, index] = (values - float(params["mean"])) / float(params["std"])
        return scaled

    def _gru_inputs(self, strokes: list[StrokeInput], raw: np.ndarray) -> tuple[torch.Tensor, ...]:
        preprocessing = self.gru_checkpoint["preprocessing"]
        categorical = preprocessing["categorical"]
        shot = [categorical["shot_type_standardized"].get(s.shot_type, 1) for s in strokes]
        hitter = [categorical["hitter"].get(s.hitter, 1) for s in strokes]
        zone = [categorical["shot_target_zone"].get(str(s.target_zone), 1) for s in strokes]
        names = [
            "player_x", "player_y", "opponent_x", "opponent_y", "delta_time_s",
            "player_movement_distance", "player_speed", "opponent_movement_distance", "opponent_speed",
        ]
        continuous = self._scale_columns(raw, names, preprocessing["continuous"])
        return (
            torch.tensor([shot], dtype=torch.long, device=self.device),
            torch.tensor([hitter], dtype=torch.long, device=self.device),
            torch.tensor([zone], dtype=torch.long, device=self.device),
            torch.tensor(continuous[None, :, :], dtype=torch.float32, device=self.device),
        )

    @staticmethod
    def _position_band(self_y: float, boundaries: list[float]) -> str:
        if self_y <= boundaries[0]:
            return "rear"
        if self_y <= boundaries[1]:
            return "middle"
        return "front"

    def _context_vector(self, previous_shot: str, previous_zone: str, band: str) -> np.ndarray:
        vocabulary = self.state_report["context_vocabulary"]
        values = {
            "previous_shot": previous_shot,
            "previous_target_zone": previous_zone,
            "self_position_band": band,
        }
        encoded: list[float] = []
        for column in ["previous_shot", "previous_target_zone", "self_position_band"]:
            categories = vocabulary[column]
            value = values[column] if values[column] in categories else "UNK"
            encoded.extend(1.0 if value == category else 0.0 for category in categories)
        return np.asarray(encoded, dtype=np.float32)

    def _cql_states(
        self, strokes: list[StrokeInput], raw: np.ndarray, hidden_before: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], list[str], list[str], list[str]]:
        observable_names = list(self.state_report["observable_features"])
        observable_rows: list[list[float]] = []
        context_vectors: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        context_keys: list[str] = []
        context_sources: list[str] = []
        previous_shot, previous_zone = "START", "START"
        boundaries = [float(value) for value in self.cql_checkpoint["config"]["self_position_boundaries_m"]]
        mask_lookup = self.state_report["context_action_masks"]
        source_lookup = self.state_report["context_mask_sources"]
        for index, stroke in enumerate(strokes):
            if stroke.hitter == "A":
                observable = [raw[index, 0], raw[index, 1], raw[index, 2], raw[index, 3], raw[index, 4],
                              raw[index, 5], raw[index, 6], raw[index, 7], raw[index, 8]]
            else:
                observable = [-raw[index, 2], 13.4 - raw[index, 3], -raw[index, 0], 13.4 - raw[index, 1], raw[index, 4],
                              raw[index, 7], raw[index, 8], raw[index, 5], raw[index, 6]]
            observable_rows.append(observable)
            band = self._position_band(float(observable[1]), boundaries)
            context_key = f"{previous_shot}|{band}"
            context_keys.append(context_key)
            context_vectors.append(self._context_vector(previous_shot, previous_zone, band))
            allowed = mask_lookup.get(context_key, list(range(len(self.actions))))
            mask = np.zeros(len(self.actions), dtype=bool)
            mask[np.asarray(allowed, dtype=int)] = True
            masks.append(mask)
            context_sources.append(str(source_lookup.get(context_key, "global")))
            previous_shot, previous_zone = stroke.shot_type, str(stroke.target_zone)

        observable_raw = np.asarray(observable_rows, dtype=np.float64)
        observable_scaled = self._scale_columns(
            observable_raw, observable_names, self.state_report["observable_preprocessing"]
        )
        states = np.concatenate(
            [hidden_before.astype(np.float32), observable_scaled, np.stack(context_vectors)], axis=1
        )
        return states, masks, context_keys, context_sources, [str(value) for value in observable_names]

    def analyze(self, request: RallyRequest) -> RallyResponse:
        strokes = request.strokes
        raw = self._raw_features(strokes)
        shot, hitter, zone, continuous = self._gru_inputs(strokes, raw)
        with torch.inference_mode():
            features = torch.cat(
                [
                    self.gru.shot_embedding(shot),
                    self.gru.hitter_embedding(hitter),
                    self.gru.zone_embedding(zone),
                    continuous,
                ],
                dim=-1,
            )
            hidden_after, _ = self.gru.gru(self.gru.input_dropout(features))
            logits = self.gru.output(hidden_after).squeeze(-1)
            probabilities = torch.sigmoid(logits)[0].cpu().numpy()
        hidden_after_np = hidden_after[0].cpu().numpy()
        hidden_before = np.zeros_like(hidden_after_np)
        hidden_before[1:] = hidden_after_np[:-1]

        states, masks, context_keys, context_sources, _ = self._cql_states(strokes, raw, hidden_before)
        with torch.inference_mode():
            q_values = self.cql(torch.tensor(states, dtype=torch.float32, device=self.device)).cpu().numpy()

        timeline: list[StrokeResult] = []
        previous_probability: float | None = None
        for index, stroke in enumerate(strokes):
            warnings: list[str] = []
            for label, position in [("선수 A", stroke.player_a_position), ("선수 B", stroke.player_b_position)]:
                if abs(position.x_m) > 2.59 or not 0.0 <= position.y_m <= 13.4:
                    warnings.append(f"{label} 기록 좌표가 코트 경계를 조금 벗어나 전처리 범위로 보정됩니다.")
            mask = masks[index]
            masked = np.where(mask, q_values[index], -np.inf)
            top_ids = np.argsort(-masked, kind="stable")[:3]
            top_actions = []
            for rank, action_id in enumerate(top_ids, start=1):
                row = self.actions.iloc[int(action_id)]
                top_actions.append(
                    RecommendedAction(
                        rank=rank,
                        shot_type=str(row["shot_type"]),
                        target_zone=int(row["target_zone"]),
                        q_value=float(q_values[index, action_id]),
                        train_count=int(row["train_count"]),
                        context_train_count=self.context_counts.get((context_keys[index], int(action_id))),
                    )
                )
            actual_label = f"{stroke.shot_type}|Z{stroke.target_zone}"
            actual_id = self.action_by_label.get(actual_label)
            actual_supported = actual_id is not None and bool(mask[actual_id])
            actual_q: float | None = None
            actual_rank: int | None = None
            if actual_supported and actual_id is not None:
                actual_q = float(q_values[index, actual_id])
                actual_rank = int(1 + np.sum(mask & (q_values[index] > q_values[index, actual_id])))
            else:
                warnings.append("선택한 행동은 이 상황의 학습 support가 부족하여 Q 순위를 비교하지 않습니다.")
            if context_sources[index] == "global":
                warnings.append("동일 맥락의 표본이 적어 전체 행동 support를 사용했습니다.")

            probability = float(probabilities[index])
            timeline.append(
                StrokeResult(
                    stroke_no=index + 1,
                    hitter=stroke.hitter,
                    shot_type=stroke.shot_type,
                    target_zone=stroke.target_zone,
                    cql=CQLResult(
                        top_actions=top_actions,
                        actual_action_supported=actual_supported,
                        actual_q=actual_q,
                        actual_rank=actual_rank,
                        context_source=context_sources[index],
                    ),
                    gru=GRUResult(
                        player_a_win_probability=probability,
                        probability_delta=(probability - previous_probability) if previous_probability is not None else None,
                    ),
                    warnings=warnings,
                )
            )
            previous_probability = probability
        return RallyResponse(
            player_a_name=request.player_a_name,
            player_b_name=request.player_b_name,
            timeline=timeline,
        )
