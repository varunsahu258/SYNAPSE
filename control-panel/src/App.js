import { useState } from 'react';

import { runExperiment, runSimulation } from './api';
import GridCanvas from './components/GridCanvas';
import InterventionTimeline from './components/InterventionTimeline';
import MetricsChart from './components/MetricsChart';
import './App.css';

const DEFAULT_FORM = {
  numAgents: 3,
  steps: 10,
  taxRate: 0.25,
  giniThreshold: 0.4,
  satisfactionThreshold: 40,
  crimeThreshold: 50,
  directorMode: 'rule_based',
};

const SLIDER_FIELDS = [
  {
    field: 'taxRate',
    id: 'taxRate',
    label: 'Tax rate',
    min: 0,
    max: 1,
    step: 0.01,
    valueLabel: (value) => `${Math.round(value * 100)}%`,
  },
  {
    field: 'giniThreshold',
    id: 'giniThreshold',
    label: 'Gini threshold',
    min: 0,
    max: 1,
    step: 0.01,
    valueLabel: (value) => value.toFixed(2),
  },
  {
    field: 'satisfactionThreshold',
    id: 'satisfactionThreshold',
    label: 'Satisfaction threshold',
    min: 0,
    max: 100,
    step: 1,
    valueLabel: (value) => `${value}`,
  },
  {
    field: 'crimeThreshold',
    id: 'crimeThreshold',
    label: 'Crime threshold',
    min: 0,
    max: 100,
    step: 1,
    valueLabel: (value) => `${value}`,
  },
  {
    field: 'numAgents',
    id: 'numAgents',
    label: 'Number of agents',
    min: 1,
    max: 50,
    step: 1,
    valueLabel: (value) => `${value} agent${value === 1 ? '' : 's'}`,
  },
  {
    field: 'steps',
    id: 'steps',
    label: 'Steps',
    min: 0,
    max: 100,
    step: 1,
    valueLabel: (value) => `${value} step${value === 1 ? '' : 's'}`,
  },
];

const EXPERIMENT_CARDS = [
  {
    key: 'no_director',
    label: 'No Director',
    accent: 'bg-blue-500',
  },
  {
    key: 'random',
    label: 'Random',
    accent: 'bg-orange-500',
  },
  {
    key: 'director_based',
    label: 'Director',
    accent: 'bg-emerald-500',
  },
];

function formatMetric(value, fallback = '—') {
  return typeof value === 'number' ? value.toFixed(2) : fallback;
}

function formatInteger(value, fallback = '—') {
  return typeof value === 'number' ? String(value) : fallback;
}

/**
 * SYNAPSES frontend control panel for running experiment variants and viewing JSON.
 */
function App() {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState('Ready');
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [simulationResult, setSimulationResult] = useState(null);
  const [uiError, setUiError] = useState('');

  function updateField(field, value) {
    setForm((current) => ({
      ...current,
      [field]: typeof current[field] === 'number' ? Number(value) : value,
    }));
  }

  async function handleRunExperiment(event) {
    event.preventDefault();
    setIsRunning(true);
    setUiError('');
    setProgress(10);
    setStatus(`Running experiment for ${form.numAgents} agents over ${form.steps} steps...`);

    try {
      if (form.directorMode === 'llm') {
        const nextSimulation = await runSimulation(form);
        setSimulationResult(nextSimulation);
        const timeline = nextSimulation.metrics_over_time ?? [];
        const final = timeline[timeline.length - 1] ?? {};
        setResult({
          experiments: {
            no_director: { metrics_over_time: [] },
            random: { metrics_over_time: [] },
            director_based: { metrics_over_time: timeline },
          },
          comparison: {
            no_director: {},
            random: {},
            director_based: {
              gini: final.gini,
              crime_rate: final.crime_rate,
              average_satisfaction: final.average_satisfaction,
              interventions: final.interventions ?? [],
            },
          },
        });
      } else {
        const nextResult = await runExperiment(form);
        setResult(nextResult);
        try {
          const nextSimulation = await runSimulation(form);
          setSimulationResult(nextSimulation);
        } catch (_simulationError) {
          setSimulationResult(null);
        }
      }
      setProgress(100);
      setStatus(`Experiment complete for ${form.numAgents} agents over ${form.steps} steps`);
    } catch (error) {
      setStatus(`Experiment failed: ${error.message}`);
    } finally {
      setIsRunning(false);
      setTimeout(() => setProgress(0), 800);
    }
  }

  const experiments = result?.experiments ?? {};
  const comparison = result?.comparison ?? {};
  const stepCount = Math.max(
    0,
    ...Object.values(experiments).map(
      (experiment) => experiment.metrics_over_time?.length ?? 0,
    ),
  );
  const interventionTimeline = (experiments.director_based?.metrics_over_time ?? []).flatMap((entry) =>
    (entry.interventions ?? []).map((item) => ({ step: entry.step, ...item })),
  );

  async function copyJson() {
    if (!result) return;
    await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setStatus('Copied result JSON to clipboard');
  }

  function downloadCsv() {
    if (!experiments.director_based?.metrics_over_time?.length) return;
    const rows = experiments.director_based.metrics_over_time;
    const header = ['step', 'gini', 'average_satisfaction', 'crime_rate', 'food_supply', 'price'];
    const body = rows.map((row) => header.map((key) => row[key]).join(',')).join('\n');
    const blob = new Blob([`${header.join(',')}\n${body}`], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'director_metrics.csv';
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,_#1d4ed8,_transparent_34%),linear-gradient(135deg,_#020617,_#0f172a_45%,_#1e293b)] px-4 py-6 text-slate-900 sm:px-6 lg:px-8">
      <div className="mx-auto grid max-w-7xl gap-6">
        <section className="overflow-hidden rounded-[2rem] border border-white/10 bg-white/10 p-6 text-white shadow-dashboard backdrop-blur md:p-8">
          <div className="grid gap-6 lg:grid-cols-[1fr_auto] lg:items-end">
            <div>
              <p className="mb-3 text-xs font-bold uppercase tracking-[0.35em] text-blue-200">SYNAPSES</p>
              <h1 className="text-4xl font-black tracking-tight sm:text-5xl">Simulation Control Panel</h1>
              <p className="mt-4 max-w-3xl text-base text-blue-100 sm:text-lg">
                Run all experiment variants through the FastAPI backend and compare their metrics in a live dashboard.
              </p>
            </div>
            <div className="grid grid-cols-3 gap-3 rounded-2xl border border-white/10 bg-slate-950/30 p-4 text-center">
              <div>
                <p className="text-2xl font-black">{form.numAgents}</p>
                <p className="text-xs font-bold uppercase tracking-wide text-blue-200">Agents</p>
              </div>
              <div>
                <p className="text-2xl font-black">{form.steps}</p>
                <p className="text-xs font-bold uppercase tracking-wide text-blue-200">Steps</p>
              </div>
              <div>
                <p className="text-2xl font-black">{Math.round(form.taxRate * 100)}%</p>
                <p className="text-xs font-bold uppercase tracking-wide text-blue-200">Tax</p>
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-[420px_1fr]">
          <form className="dashboard-card grid gap-6 self-start" onSubmit={handleRunExperiment}>
            <div>
              <p className="text-sm font-bold uppercase tracking-[0.2em] text-blue-600">Configure</p>
              <h2 className="mt-1 text-2xl font-black text-slate-950">Experiment controls</h2>
              <p className="mt-2 text-sm font-medium text-slate-500">
                Tune population size, duration, and redistribution before comparing variants.
              </p>
            </div>

            <div className="grid gap-5">
              {SLIDER_FIELDS.map(({ field, id, label, min, max, step, valueLabel }) => (
                <div className="grid gap-2" key={field}>
                  <div className="flex items-center justify-between gap-3">
                    <label className="text-sm font-bold text-slate-700" htmlFor={id}>{label}</label>
                    <output className="slider-value" htmlFor={id}>{valueLabel(form[field])}</output>
                  </div>
                  <input
                    className="h-2 w-full rounded-lg bg-slate-200"
                    id={id}
                    max={max}
                    min={min}
                    step={step}
                    type="range"
                    value={form[field]}
                    onChange={(event) => updateField(field, event.target.value)}
                  />
                  <span className="flex justify-between text-xs font-bold text-slate-400" aria-hidden="true">
                    <span>{min}</span>
                    <span>{max}</span>
                  </span>
                </div>
              ))}
              <label className="flex items-center gap-2 text-sm font-bold text-slate-700">
                <input
                  checked={form.directorMode === 'llm'}
                  type="checkbox"
                  onChange={(event) => updateField('directorMode', event.target.checked ? 'llm' : 'rule_based')}
                />
                Use LLM Director
              </label>
              {form.directorMode === 'llm' && (
                <div className="grid gap-2 rounded-xl border border-amber-200 bg-amber-50 p-3">
                  <p className="text-xs font-black text-amber-800">Using NVIDIA Nemotron via OpenRouter</p>
                  <p className="text-xs font-semibold text-amber-700">LLM mode uses the server-side OpenRouter key configured by the API operator.</p>
                </div>
              )}

            </div>

            <div className="rounded-2xl bg-slate-100 p-4">
              <p className="text-sm font-bold text-slate-600">
                Experiment request: {form.numAgents} agents, {form.steps} steps, {Math.round(form.taxRate * 100)}% tax
              </p>
              <button
                className="mt-4 w-full rounded-2xl bg-blue-600 px-5 py-3 font-black text-white shadow-lg shadow-blue-600/30 transition hover:-translate-y-0.5 hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:translate-y-0"
                disabled={isRunning}
                type="submit"
              >
                {isRunning ? 'Running...' : 'Run Experiment'}
              </button>
              <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-slate-200">
                <div className="h-full bg-blue-500 transition-all" style={{ width: `${progress}%` }} />
              </div>
            </div>
          </form>

          <section className="grid gap-6">
            <div className="grid gap-4 sm:grid-cols-3">
              {EXPERIMENT_CARDS.map(({ key, label, accent }) => {
                const finalMetrics = comparison[key] ?? {};

                return (
                  <article className="stat-card" key={key}>
                    <div className="mb-4 flex items-center gap-3">
                      <span className={`h-3 w-3 rounded-full ${accent}`} />
                      <h2 className="text-base font-black text-slate-950">{label}</h2>
                    </div>
                    <dl className="grid grid-cols-3 gap-3 text-center">
                      <div>
                        <dt className="text-[0.65rem] font-black uppercase tracking-wide text-slate-400">Gini</dt>
                        <dd className="mt-1 text-xl font-black text-slate-950">{formatMetric(finalMetrics.gini)}</dd>
                      </div>
                      <div>
                        <dt className="text-[0.65rem] font-black uppercase tracking-wide text-slate-400">Crime</dt>
                        <dd className="mt-1 text-xl font-black text-slate-950">{formatInteger(finalMetrics.crime_rate)}</dd>
                      </div>
                      <div>
                        <dt className="text-[0.65rem] font-black uppercase tracking-wide text-slate-400">Sat.</dt>
                        <dd className="mt-1 text-xl font-black text-slate-950">{formatMetric(finalMetrics.average_satisfaction)}</dd>
                      </div>
                    </dl>
                  </article>
                );
              })}
            </div>

            <p className="rounded-2xl border border-blue-100 bg-blue-50 px-4 py-3 text-sm font-bold text-blue-800" role="status">
              {status}
            </p>
            {uiError && <p className="rounded-xl border border-red-200 bg-red-50 px-4 py-2 text-sm font-bold text-red-700">{uiError}</p>}

            <section className="grid gap-6 2xl:grid-cols-[1fr_420px]">
              <MetricsChart experiments={experiments} />

              <div className="grid gap-6">
                <GridCanvas gridState={simulationResult?.grid_state} />
                <InterventionTimeline timeline={interventionTimeline} />
              </div>
            </section>

            <article className="dashboard-card min-w-0">
                <div className="mb-4 flex items-center justify-between gap-4">
                  <h2 className="text-xl font-black text-slate-950">JSON result</h2>
                  <div className="flex items-center gap-2">
                    <button className="rounded-lg bg-slate-100 px-3 py-1 text-xs font-black text-slate-700" onClick={downloadCsv} type="button">Download CSV</button>
                    <button className="rounded-lg bg-slate-100 px-3 py-1 text-xs font-black text-slate-700" onClick={copyJson} type="button">Copy JSON</button>
                    <span className="rounded-full bg-slate-100 px-3 py-1 text-sm font-black text-slate-500">{stepCount} steps</span>
                  </div>
                </div>
                <pre className="max-h-[720px] min-h-80 overflow-auto rounded-2xl bg-slate-950 p-4 text-sm leading-6 text-blue-100 shadow-inner">
                  {result ? JSON.stringify(result, null, 2) : 'No experiment result yet.'}
                </pre>
              </article>
          </section>
        </section>
      </div>
    </main>
  );
}

export default App;
