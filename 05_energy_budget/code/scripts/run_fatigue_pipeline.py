"""Reproducible six-phase fatigue-terminal research pipeline.

--resume reuses frozen fits and held-out evaluation; it never retunes on test.
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
import hashlib
from dataclasses import asdict
import time
import numpy as np
import pandas as pd

from src.fatigue_baseline import (Baseline, baseline_events, select_history, select_role_tau, fit_baseline,
                                  baseline_metrics, export_baseline)
from src.reconstruct_rally_energy import write_energy_tables, audit_split_separation
from src.fit_fatigue_terminal import (FatigueModel, fit_cohort, fit_selected_pair, predictions,
    three_class_metrics, binary_metrics, coefficient_table, bootstrap_models, fit_hazard, context_offsets, domain)
from src.build_fatigue_transition import FatigueTransition, event_joint_metrics, export_transition_samples
from src.solve_stamina_value import solve_stamina_value, monte_carlo_value, StaminaGrid
from src.fatigue_reporting import calibration_table, energy_quantile_table, create_figures, write_report
from src.estimate_markov_q import N_STATES


def log(message):
    print(message, flush=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def phase(root, number, message, statistics=None):
    path = root / "phase_status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    implementation = {1:["src/reconstruct_rally_energy.py"],2:["src/reconstruct_rally_energy.py"],
                      3:["src/fit_fatigue_terminal.py","src/fatigue_baseline.py"],4:["src/fit_fatigue_terminal.py"],
                      5:["src/build_fatigue_transition.py"],6:["src/solve_stamina_value.py","scripts/run_fatigue_pipeline.py"]}
    artifacts = {1:["terminal_event_audit.csv","energy_audit.json"],2:["event_energy_table.parquet"],
                 3:["model_selection.csv","shot_support.csv","models"],4:["bootstrap_coefficients.parquet","bootstrap_summary.csv","control_model_coefficients.csv"],
                 5:["fatigue_terminal_samples.parquet","fatigue_q_samples.parquet"],6:["stamina","stamina_value_summary.parquet","stamina_numerical_validation.csv"]}
    status[str(number)] = dict(message=message, statistics=statistics or {},
                               implementation_files=implementation[number],artifacts=artifacts[number])
    save_json(path, status)
    log(f"PHASE {number}: {message}; {json.dumps(statistics or {})}")


def load_cohort(tables, cohort, root, selection, support):
    model = FatigueModel.load(root / "models" / f"{cohort}_development")
    diagnostic = FatigueModel.load(root / "models" / f"{cohort}_diagnostic_development")
    train = tables["train"][tables["train"][f"fit_{cohort}_eligible"]].reset_index(drop=True)
    validation = tables["validation"][tables["validation"][f"fit_{cohort}_eligible"]].reset_index(drop=True)
    cached_calibration = {}
    if (root / "fatigue_fit_diagnostics.json").exists():
        cached_calibration = json.loads((root / "fatigue_fit_diagnostics.json").read_text()).get("calibration_selection", {}).get(cohort, {})
    return dict(model=model, train=train, validation=validation,
                family_models={f: FatigueModel.load(root / "models" / f"{cohort}_{f}_development") for f in ("M0","M1","M2")},
                diagnostic=dict(e=diagnostic.e_model, u=diagnostic.u_model),
                selection=selection[selection.cohort.eq(cohort)], support=support[support.cohort.eq(cohort)],
                calibration=cached_calibration or dict(selected=bool(model.e_model.calibration_shift or model.u_model.calibration_shift),
                                                       shift_e=model.e_model.calibration_shift, shift_u=model.u_model.calibration_shift))


def evaluate_locked_models(cohorts, tables, baseline, root, fingerprint):
    cache = root / "held_out_evaluation.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached["fingerprint"] == fingerprint:
            log("Reusing frozen held-out evaluation; test is not evaluated again.")
            return cached["metrics"], pd.read_parquet(root / "fatigue_model_predictions.parquet")
        raise ValueError("held-out evaluation already exists for different inputs/settings; use a new output directory")
    metrics, frames = [], []
    for cohort, result in cohorts.items():
        models = dict(result["family_models"], selected=result["model"])
        for split, table in tables.items():
            events = table[table[f"fit_{cohort}_eligible"]].reset_index(drop=True)
            for label, model in models.items():
                prediction = predictions(model, events)
                stats = three_class_metrics(events, prediction)
                stats.update(event_joint_metrics(FatigueTransition(baseline, model), events))
                for endpoint in ("e","u"):
                    mask = np.ones(len(events), bool) if endpoint == "e" else ~events.event_class.eq("E").to_numpy()
                    binary = binary_metrics(events.event_class.eq("E" if endpoint == "e" else "U").to_numpy()[mask], prediction[endpoint][mask])
                    stats.update({f"{endpoint}_{k}":v for k,v in binary.items()})
                metrics.append(dict(cohort=cohort, split=split, model=label, **stats))
                if label == "selected":
                    columns = ["split","match_id","set_id","rally_id","stroke_no","state_id","parity","shot_type","action_index",
                               "event_class","classification_reliability","cumulative_energy_A_before","cumulative_energy_B_before",
                               "cumulative_energy_hitter_before","cumulative_energy_receiver_before"]
                    frame = events[columns].copy()
                    for key in ("e","u","E","U","C","extrapolation"):
                        frame[key] = prediction[key]
                    frame["cohort"], frame["fit_scope"] = cohort, "train"
                    frames.append(frame)
    predictions_table = pd.concat(frames, ignore_index=True)
    predictions_table.to_parquet(root / "fatigue_model_predictions.parquet", index=False)
    pd.DataFrame(metrics).to_csv(root / "held_out_metrics.csv", index=False)
    save_json(cache, dict(fingerprint=fingerprint, test_used_for_tuning=False, metrics=metrics))
    return metrics, predictions_table


def marginal_terminal_comparison(cohorts, tables, baseline):
    rows = []
    for cohort, result in cohorts.items():
        for split, table in tables.items():
            events = table[table[f"fit_{cohort}_eligible"]]
            p = predictions(result["model"], events)
            r, i, s = events.parity.to_numpy(dtype=int), events.state_id.to_numpy(dtype=int), events.action_index.to_numpy(dtype=int)
            base = baseline.probability[r,i,s]
            pi = base.sum(axis=-1)
            base_e = np.where(r==0,base[:,16],base[:,17])/pi
            base_u = np.where(r==0,base[:,17],base[:,16])/pi
            base_c = base[:,:16].sum(axis=-1)/pi
            for label, old in (("E",base_e),("U",base_u),("C",base_c)):
                rows.append(dict(cohort=cohort,split=split,event=label,events=len(events),
                                 observed_rate=float(events.event_class.eq(label).mean()),
                                 model_mean_over_observed_consumption=float(p[label].mean()),
                                 baseline_fatigue_mixed_mean=float(old.mean())))
    return pd.DataFrame(rows)


def run_stamina(system, train_table, root, grid_intervals, mc_samples, resume, max_grid_intervals=160):
    if max_grid_intervals < grid_intervals:
        raise ValueError("maximum grid size must be at least the requested starting grid")
    observed = train_table[train_table.terminal.eq(1) &
                           ~train_table.cumulative_energy_A_contains_imputation & ~train_table.cumulative_energy_B_contains_imputation &
                           ~train_table.rally_frame_chronology_uncertain]
    energy = observed[["cumulative_energy_A_before","cumulative_energy_B_before"]].to_numpy().ravel()
    energy = energy[np.isfinite(energy)]
    budgets = np.quantile(energy,[.5,.75,.95])
    representative = []
    eligible = train_table[train_table.fit_strict_eligible]
    for parity in (0,1):
        for state in eligible[eligible.parity.eq(parity)].state_id.value_counts().head(4).index:
            representative.append((int(state),parity))
    scenario_summaries, initial_tables, validation_rows = [], [], []
    directory = root / "stamina"
    directory.mkdir(exist_ok=True)
    source_digest = hashlib.sha256()
    for array in (system.pi, system.T, system.cost_a, system.cost_b,
                  system.model.e_model.theta, system.model.u_model.theta,
                  system.model.e_model.offset, system.model.u_model.offset):
        source_digest.update(np.ascontiguousarray(array).tobytes())
    signature = dict(model_sha256=source_digest.hexdigest(), d_star=system.model.d_star,
                     requested_grid_intervals=grid_intervals,mc_samples=mc_samples,terminal_costs={},
                     initial_budgets=budgets.tolist(),solver="bilinear_cell_fixed_point_v1")
    protocol = directory / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != signature:
        raise ValueError("stamina model/budget/grid/MC settings changed; use a new output directory")
    if not protocol.exists():
        # Existing completed scenarios must be compatible before adopting a cache.
        for existing in directory.glob("scenario_p*.json"):
            cached = json.loads(existing.read_text())
            requested = cached.get("requested_grid_intervals", cached["grid_intervals"]//(2 if cached["refined"] else 1))
            if requested != grid_intervals:
                raise ValueError("cached stamina grid settings differ")
            validation = pd.read_csv(directory / f"validation_p{cached['percentile']}.csv")
            if not validation.mc_samples.eq(mc_samples).all():
                raise ValueError("cached Monte Carlo sample settings differ")
        save_json(protocol,signature)
    for scenario, (percentile, budget) in enumerate(zip((50,75,95),budgets)):
        metadata_path = directory / f"scenario_p{percentile}.json"
        if resume and metadata_path.exists():
            summary = json.loads(metadata_path.read_text())
            if summary["numerical_validation_passed"] or summary["grid_intervals"] >= max_grid_intervals:
                scenario_summaries.append(summary)
                initial_tables.append(pd.read_parquet(directory / f"initial_p{percentile}.parquet"))
                validation_rows.extend(pd.read_csv(directory / f"validation_p{percentile}.csv").to_dict("records"))
                log(f"Reused completed stamina scenario p{percentile}")
                continue
            log(f"Refining cached unresolved stamina p{percentile} from {summary['grid_intervals']} intervals")
            fine = StaminaGrid.load(directory / f"scenario_p{percentile}.npz")
            used_intervals = int(summary["grid_intervals"])
            comparisons = pd.read_csv(directory / f"validation_p{percentile}.csv").to_dict("records")
            mc_cache = [(int(row["parity"])*N_STATES+int(row["state_id"]),float(budget*row["remaining_fraction"]),
                         {key[3:]:value for key,value in row.items() if key.startswith("mc_")}) for row in comparisons]
            history = summary.get("refinement_history",[dict(intervals=used_intervals,
                                  max_refinement_difference=summary["max_refinement_difference"],
                                  max_mc_excess_95ci=summary["max_mc_difference_beyond_95ci"])])
        else:
            log(f"Solving stamina scenario p{percentile}: initial A=B={budget:.3f} J/kg")
            coarse = solve_stamina_value(system,float(budget),float(budget),max(2,grid_intervals//2),progress=log)
            fine = solve_stamina_value(system,float(budget),float(budget),grid_intervals,progress=log)
            fractions = (1., .75, .5, .25) if percentile == 95 else (1.,)
            comparisons,mc_cache = [],[]
            for state,parity in representative:
                for fraction in fractions:
                    remaining = float(budget*fraction)
                    mc = monte_carlo_value(system,state,parity,float(budget),float(budget),remaining,remaining,
                                           samples=mc_samples,seed=42+scenario*100+len(comparisons))
                    aug = parity*N_STATES+state
                    small = coarse.query(aug,remaining,remaining)
                    large = fine.query(aug,remaining,remaining)
                    row = dict(percentile=percentile,state_id=state,parity=parity,remaining_fraction=fraction,
                               grid_intervals=grid_intervals,coarse_midpoint=float(small["V_midpoint"]),
                               grid_midpoint=float(large["V_midpoint"]),grid_lower=float(large["V_lower"]),grid_upper=float(large["V_upper"]),
                               refinement_difference=max(abs(float(small[k]-large[k])) for k in ("V_lower","V_midpoint","V_upper")),
                               mc_difference=max(abs(float(large[k])-mc[k]) for k in ("V_lower","V_midpoint","V_upper")),
                               mc_excess_95ci=max(abs(float(large[k])-mc[k])-1.96*mc[se] for k,se in
                                                 (("V_lower","lower_standard_error"),("V_midpoint","midpoint_standard_error"),("V_upper","upper_standard_error"))),
                               **{f"mc_{k}":v for k,v in mc.items()})
                    comparisons.append(row)
                    mc_cache.append((aug,remaining,mc))
            del coarse
            used_intervals=grid_intervals
            history=[]
        max_refinement = max(r["refinement_difference"] for r in comparisons)
        max_mc_excess = max(r["mc_excess_95ci"] for r in comparisons)
        while (max_refinement>.02 or max_mc_excess>.02) and used_intervals<max_grid_intervals:
            record = dict(intervals=used_intervals,max_refinement_difference=max_refinement,max_mc_excess_95ci=max_mc_excess)
            if not history or history[-1] != record:
                history.append(record)
            next_intervals=min(used_intervals*2,max_grid_intervals)
            log(f"p{percentile}: refinement/MC excess {max_refinement:.4f}/{max_mc_excess:.4f}; refining to {next_intervals}")
            previous_queries=[fine.query(aug,remaining,remaining) for aug,remaining,mc in mc_cache]
            del fine
            refined_grid = solve_stamina_value(system,float(budget),float(budget),next_intervals,progress=log)
            for row,(aug,remaining,mc),previous in zip(comparisons,mc_cache,previous_queries):
                current = refined_grid.query(aug,remaining,remaining)
                row.update(grid_intervals=next_intervals,coarse_midpoint=float(previous["V_midpoint"]),
                           grid_midpoint=float(current["V_midpoint"]),grid_lower=float(current["V_lower"]),grid_upper=float(current["V_upper"]),
                           refinement_difference=max(abs(float(previous[k]-current[k])) for k in ("V_lower","V_midpoint","V_upper")),
                           mc_difference=max(abs(float(current[k])-mc[k]) for k in ("V_lower","V_midpoint","V_upper")),
                           mc_excess_95ci=max(abs(float(current[k])-mc[k])-1.96*mc[se] for k,se in
                                             (("V_lower","lower_standard_error"),("V_midpoint","midpoint_standard_error"),("V_upper","upper_standard_error"))))
            fine = refined_grid
            used_intervals = next_intervals
            max_refinement = max(r["refinement_difference"] for r in comparisons)
            max_mc_excess = max(r["mc_excess_95ci"] for r in comparisons)
        fine.save(directory / f"scenario_p{percentile}.npz")
        result = fine.query(np.arange(2*N_STATES),float(budget),float(budget))
        initial = pd.DataFrame(dict(percentile=percentile,initial_budget_A=budget,initial_budget_B=budget,
                                    augmented_state=np.arange(2*N_STATES),parity=np.repeat(np.arange(2),N_STATES),
                                    state_id=np.tile(np.arange(N_STATES),2), **result))
        initial.to_parquet(directory / f"initial_p{percentile}.parquet",index=False)
        pd.DataFrame(comparisons).to_csv(directory / f"validation_p{percentile}.csv",index=False)
        summary = dict(percentile=percentile,initial_budget_A=float(budget),initial_budget_B=float(budget),
                       grid_intervals=used_intervals,requested_grid_intervals=grid_intervals,max_bellman_residual=fine.max_residual,
                       max_cell_iterations=fine.max_iterations,refined=used_intervals>grid_intervals,refinement_history=history,
                       max_refinement_difference=max_refinement,max_mc_difference_beyond_95ci=max_mc_excess,
                       numerical_validation_passed=max_refinement<=.02 and max_mc_excess<=.02,
                       initial_midpoint_mean=float(initial.V_midpoint.mean()),
                       initial_lower_mean=float(initial.V_lower.mean()),initial_upper_mean=float(initial.V_upper.mean()),
                       initial_ambiguity_mean=float(initial.simultaneous_exhaustion_probability.mean()),
                       initial_parity_complement_max_error=float(np.max(np.abs(initial.V_midpoint.to_numpy().reshape(2,N_STATES).sum(axis=0)-1))))
        save_json(metadata_path,summary)
        scenario_summaries.append(summary)
        initial_tables.append(initial)
        validation_rows.extend(comparisons)
        del fine
        phase(root,6,f"Completed stamina p{percentile}",summary)
    pd.concat(initial_tables,ignore_index=True).to_parquet(root / "stamina_value_summary.parquet",index=False)
    pd.DataFrame(validation_rows).to_csv(root / "stamina_numerical_validation.csv",index=False)
    phase(root,6,"All stamina scenarios completed",dict(
        scenarios=scenario_summaries,
        all_numerical_checks_passed=all(s["numerical_validation_passed"] for s in scenario_summaries)))
    return dict(scenarios=scenario_summaries,terminal_additional_cost_assumption="zero, explicitly unobserved",
                simultaneous_exhaustion="A-loss/A-win bounds; midpoint 0.5 sensitivity",
                monte_carlo_samples_per_query=mc_samples,observed_train_rally_energy_population=len(energy),
                all_numerical_checks_passed=all(s["numerical_validation_passed"] for s in scenario_summaries))


def run(args):
    start = time.perf_counter()
    root = args.output_root
    root.mkdir(parents=True,exist_ok=True)
    split_paths = {name:args.split_root / f"{name}.csv" for name in ("train","validation","test")}
    input_hashes = {name:digest(path) for name,path in split_paths.items()}
    protocol_path = root / "input_protocol.json"
    protocol = dict(input_sha256=input_hashes,bootstrap_repeats=args.bootstrap_repeats,seed=42,
                    chronology_policy="exclude_uncertain_frame_rallies_from_hazard_and_cost_fits",
                    fit_protocol="train fit; validation selection; frozen test; train+validation export")
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("input/protocol changed; use a new output directory")
    else:
        save_json(protocol_path,protocol)
    frames = {name:pd.read_csv(path) for name,path in split_paths.items()}
    audit_split_separation(frames)
    event_frames = {name:baseline_events(frame,name) for name,frame in frames.items()}
    if args.resume and (root / "baseline_train.npz").exists():
        development = Baseline.load(root / "baseline_train.npz")
        baseline_summary = json.loads((root / "baseline_selection.json").read_text())
    else:
        order,alphas,summary,trials = select_history(event_frames["train"],event_frames["validation"])
        tau,tau_table = select_role_tau(event_frames["train"],event_frames["validation"],order)
        development = fit_baseline(event_frames["train"],order,alphas,tau)
        development.save(root / "baseline_train.npz")
        summary.to_csv(root / "history_comparison.csv",index=False)
        trials.to_csv(root / "validation_alpha_grid.csv",index=False)
        tau_table.to_csv(root / "validation_tau_grid.csv",index=False)
        baseline_summary = dict(history_order=order,alphas=alphas,tau=tau,
                                validation=baseline_metrics(development,event_frames["validation"]),
                                alpha_on_grid_boundary=any(a in (.5,5000.) for a in alphas))
        save_json(root / "baseline_selection.json",baseline_summary)
    log(f"BASELINE: {json.dumps(baseline_summary)}")
    if args.resume and (root / "event_energy_table.parquet").exists():
        table = pd.read_parquet(root / "event_energy_table.parquet")
        tables = {name:table[table.split.eq(name)].reset_index(drop=True) for name in frames}
        energy_audit = json.loads((root / "energy_audit.json").read_text())
    else:
        tables,energy_audit = write_energy_tables(frames,development,root,args.raw_root)
    phase(root,1,"Raw/terminal/split audit completed",energy_audit)
    phase(root,2,"Pre-shot energy reconstructed; observed/imputed intervals labeled",dict(summaries=energy_audit["summaries"]))
    if args.resume and (root / "model_selection.csv").exists():
        selection,support = pd.read_csv(root / "model_selection.csv"),pd.read_csv(root / "shot_support.csv")
        cohorts = {name:load_cohort(tables,name,root,selection,support) for name in ("strict","proxy")}
    else:
        cohorts = {}
        for cohort in ("strict","proxy"):
            log(f"Fitting {cohort} terminal models")
            cohorts[cohort] = fit_cohort(tables,cohort,development.order,development.alphas,root,
                                       d_star=None if cohort=="strict" else cohorts["strict"]["model"].d_star,progress=log)
        pd.concat([v["selection"] for v in cohorts.values()],ignore_index=True).to_csv(root / "model_selection.csv",index=False)
        pd.concat([v["support"] for v in cohorts.values()],ignore_index=True).to_csv(root / "shot_support.csv",index=False)
    selected_specs = {c:dict(e=asdict(r["model"].e_model.spec),u=asdict(r["model"].u_model.spec)) for c,r in cohorts.items()}
    phase(root,3,"M0/M1/M2 models fitted and validation selection frozen",selected_specs)
    strict = cohorts["strict"]
    if args.resume and (root / "bootstrap_summary.csv").exists():
        bootstrap = pd.read_csv(root / "bootstrap_summary.csv")
    else:
        models = dict(selected_e=strict["model"].e_model,selected_u=strict["model"].u_model,
                      diagnostic_e=strict["diagnostic"]["e"],diagnostic_u=strict["diagnostic"]["u"])
        samples,bootstrap = bootstrap_models(strict["train"],models,development.order,development.alphas,
                                            strict["model"].d_star,args.bootstrap_repeats,progress=log)
        samples.to_parquet(root / "bootstrap_coefficients.parquet",index=False)
        bootstrap.to_csv(root / "bootstrap_summary.csv",index=False)
    control_rows = []
    offsets = context_offsets(strict["train"],development.order,development.alphas)
    for index,endpoint in enumerate(("e","u")):
        diagnostic = strict["diagnostic"][endpoint]
        control = fit_hazard(strict["train"],offsets[index],diagnostic.spec,strict["model"].d_star,controls=True)
        table = coefficient_table(control,f"controlled_{endpoint}",False)
        control_rows.append(table)
    pd.concat(control_rows).to_csv(root / "control_model_coefficients.csv",index=False)
    phase(root,4,"Match bootstrap and nuisance-control sensitivity completed",dict(bootstrap_repeats=args.bootstrap_repeats))
    fingerprint = hashlib.sha256(json.dumps(dict(input_hashes=input_hashes,selected_specs=selected_specs,
                                                model_sha256={c:digest(root / "models" / f"{c}_development" / "parameters.npz") for c in cohorts}),sort_keys=True).encode()).hexdigest()
    metrics,prediction_table = evaluate_locked_models(cohorts,tables,development,root,fingerprint)
    marginal_terminal_comparison(cohorts,tables,development).to_csv(root / "marginal_terminal_comparison.csv",index=False)
    coefficient_frames = []
    for cohort,result in cohorts.items():
        coefficient_frames.append(coefficient_table(result["model"],f"{cohort}_development"))
    for endpoint,diagnostic in strict["diagnostic"].items():
        coefficient_frames.append(coefficient_table(diagnostic,f"diagnostic_{endpoint}",False))
    final_events = pd.concat([event_frames["train"],event_frames["validation"]],ignore_index=True)
    final_baseline = fit_baseline(final_events,development.order,development.alphas,development.tau,"train_validation")
    final_baseline.save(root / "baseline_final.npz")
    export_baseline(final_baseline,final_events,args.baseline_output_root,dict(selection=baseline_summary,input_sha256=input_hashes,
                    fit_protocol="train selection using validation; final export train+validation; test never used for tuning"))
    for cohort,result in cohorts.items():
        final = fit_selected_pair(pd.concat([result["train"],result["validation"]],ignore_index=True),result["model"],"train_validation")
        final.save(root / "models" / f"{cohort}_final")
        result["final"] = final
        coefficient_frames.append(coefficient_table(final,f"{cohort}_final"))
    coefficients = pd.concat(coefficient_frames,ignore_index=True)
    coefficients["bootstrap_model"] = np.where(coefficients.model.eq("strict_development"),
                                               "selected_"+coefficients.endpoint,coefficients.model)
    coefficients = coefficients.merge(bootstrap.rename(columns={"model":"bootstrap_model"}),how="left",
                                       on=["bootstrap_model","endpoint","term"])
    coefficients.to_csv(root / "fatigue_model_coefficients.csv",index=False)
    system = FatigueTransition(final_baseline,strict["final"])
    transition_checks = export_transition_samples(system,root)
    phase(root,5,"Fatigue-dependent w/l/Q exported; probability and static reconstruction checked",transition_checks)
    calibration = calibration_table(prediction_table)
    calibration.to_csv(root / "calibration_bins.csv",index=False)
    quantiles = energy_quantile_table(prediction_table)
    quantiles.to_csv(root / "energy_quantile_rates.csv",index=False)
    diagnostic_pair = FatigueModel.load(root / "models/strict_diagnostic_development")
    create_figures(root,strict["model"],strict["train"],calibration,quantiles,bootstrap,diagnostic_pair)
    diagnostics = dict(input_sha256=input_hashes,energy_audit=energy_audit,baseline=baseline_summary,d_star=strict["model"].d_star,
                       selected_specs=selected_specs,bootstrap_repeats=args.bootstrap_repeats,held_out_metrics=metrics,
                       test_used_for_tuning=False,development_fit_scope="train",final_fit_scope="train_validation",
                       energy_correlation=float(strict["train"][["cumulative_energy_hitter_before","cumulative_energy_receiver_before"]].corr().iloc[0,1]),
                       calibration_selection={c:r["calibration"] for c,r in cohorts.items()},transition_checks=transition_checks)
    diagnostics["source_sha256"] = {str(path.relative_to(ROOT)):digest(path) for path in
                                     [*sorted((ROOT/"src").glob("*fatigue*.py")),ROOT/"src/reconstruct_rally_energy.py",
                                      ROOT/"src/solve_stamina_value.py",ROOT/"scripts/run_fatigue_pipeline.py",ROOT/"requirements.txt"]}
    diagnostics["debug_evaluation_before_chronology_audit"] = dict(
        occurred=(ROOT/"outputs/fatigue_terminal_before_frame_audit/held_out_evaluation.json").exists(),
        archive="outputs/fatigue_terminal_before_frame_audit",
        interpretation="Initial debug evaluation preceded the frame-quality audit; test is not pristine. No test scores used to choose the corrected rule or model.")
    save_json(root / "fatigue_fit_diagnostics.json",diagnostics)
    write_report(root,diagnostics)
    if not args.skip_stamina:
        diagnostics["stamina"] = run_stamina(system,tables["train"],root,args.grid_intervals,args.mc_samples,args.resume,args.max_grid_intervals)
    diagnostics["elapsed_seconds"] = time.perf_counter()-start
    save_json(root / "fatigue_fit_diagnostics.json",diagnostics)
    write_report(root,diagnostics)
    log(f"DONE: {root.resolve()}; elapsed {diagnostics['elapsed_seconds']:.1f}s")
    return diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root",type=Path,default=ROOT / "data/splits")
    parser.add_argument("--raw-root",type=Path,default=ROOT / "raw")
    parser.add_argument("--output-root",type=Path,default=ROOT / "outputs/fatigue_terminal")
    parser.add_argument("--baseline-output-root",type=Path,default=ROOT / "outputs/markov_parity_pooled")
    parser.add_argument("--bootstrap-repeats",type=int,default=200)
    parser.add_argument("--grid-intervals",type=int,default=40)
    parser.add_argument("--max-grid-intervals",type=int,default=160)
    parser.add_argument("--mc-samples",type=int,default=20000)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--skip-stamina",action="store_true",help="fit/transition development runs only; default includes V")
    args = parser.parse_args()
    run(args)
