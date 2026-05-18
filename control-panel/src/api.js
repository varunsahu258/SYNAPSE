import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';

const client = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

function simulationPayload({
  numAgents,
  steps,
  taxRate,
  giniThreshold,
  satisfactionThreshold,
  crimeThreshold,
  directorMode,
}) {
  return {
    num_agents: numAgents,
    steps,
    tax_rate: taxRate,
    gini_threshold: giniThreshold,
    satisfaction_threshold: satisfactionThreshold,
    crime_threshold: crimeThreshold,
    director_mode: directorMode,
  };
}

/**
 * Run a SYNAPSES simulation through the FastAPI backend.
 * @param {{numAgents: number, steps: number, taxRate: number}} config
 * @returns {Promise<object>} full JSON response from /run_simulation
 */
export async function runSimulation(config) {
  const response = await client.post('/run_simulation', simulationPayload(config));

  return response.data;
}

/**
 * Run all SYNAPSES experiment variants through the FastAPI backend.
 * @param {{numAgents: number, steps: number, taxRate: number}} config
 * @returns {Promise<object>} full JSON response from /run_experiment
 */
export async function runExperiment(config) {
  const response = await client.post('/run_experiment', simulationPayload(config));

  return response.data;
}
