import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';

import App from './App';
import { runExperiment, runSimulation } from './api';

jest.mock('./api', () => ({
  runExperiment: jest.fn(),
  runSimulation: jest.fn(),
}));

const EXPERIMENT_RESPONSE = {
  experiments: {
    no_director: {
      metrics_over_time: [
        { step: 1, gini: 0.12, crime_rate: 18 },
        { step: 2, gini: 0.18, crime_rate: 24 },
      ],
    },
    random: {
      metrics_over_time: [
        { step: 1, gini: 0.1, crime_rate: 16 },
        { step: 2, gini: 0.14, crime_rate: 20 },
      ],
    },
    director_based: {
      metrics_over_time: [
        { step: 1, gini: 0.08, crime_rate: 14 },
        { step: 2, gini: 0.11, crime_rate: 17 },
      ],
    },
  },
  comparison: {},
};

afterEach(() => {
  jest.clearAllMocks();
});

test('renders the run experiment control panel', () => {
  render(<App />);
  expect(screen.getByText(/Simulation Control Panel/i)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Run Experiment/i })).toBeInTheDocument();
  expect(screen.getByLabelText(/Number of agents/i)).toHaveAttribute('type', 'range');
  expect(screen.getByLabelText(/Steps/i)).toHaveAttribute('type', 'range');
  expect(screen.getByRole('heading', { name: /JSON result/i })).toBeInTheDocument();
});

test('sends slider values to the experiment API', async () => {
  runExperiment.mockResolvedValue({ experiments: {}, comparison: {} });
  runSimulation.mockResolvedValue({ metrics_over_time: [], grid_state: { width: 1, height: 1, agents: [] } });

  render(<App />);

  fireEvent.change(screen.getByLabelText(/Number of agents/i), { target: { value: '12' } });
  fireEvent.change(screen.getByLabelText(/Steps/i), { target: { value: '24' } });
  fireEvent.click(screen.getByRole('button', { name: /Run Experiment/i }));

  await waitFor(() => {
    expect(runExperiment).toHaveBeenCalledWith({
      numAgents: 12,
      steps: 24,
      taxRate: 0.25,
      giniThreshold: 0.4,
      satisfactionThreshold: 40,
      crimeThreshold: 50,
      directorMode: 'rule_based',
    });
  });
  expect(
    await screen.findByText(/Experiment complete for 12 agents over 24 steps/i),
  ).toBeInTheDocument();
});

test('plots experiment lines from the API response with labels and legends', async () => {
  runExperiment.mockResolvedValue(EXPERIMENT_RESPONSE);
  runSimulation.mockResolvedValue({ metrics_over_time: [], grid_state: { width: 1, height: 1, agents: [] } });

  render(<App />);

  fireEvent.click(screen.getByRole('button', { name: /Run Experiment/i }));

  expect(await screen.findByRole('heading', { name: /Gini over time/i })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: /Crime rate over time/i })).toBeInTheDocument();
  const legends = screen.getAllByRole('list', { name: /legend/i });
  expect(legends).toHaveLength(2);
  legends.forEach((legend) => {
    expect(within(legend).getByText(/No Director/i)).toBeInTheDocument();
    expect(within(legend).getByText(/Random/i)).toBeInTheDocument();
    expect(within(legend).getByText(/^Director$/i)).toBeInTheDocument();
  });
  expect(screen.getByText(/2 steps/i)).toBeInTheDocument();
});
