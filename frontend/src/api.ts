/* API client + shared types — mirrors backend/app/schemas/api.py */

export interface SearchFilters {
  sex?: "male" | "female" | null;
  outcome_class?: "improved" | "deteriorated" | "deceased" | "unknown" | null;
  age_min?: number | null;
  age_max?: number | null;
}

export interface CaseResult {
  case_id: string;
  score: number;
  sex: string;
  age: number | null;
  age_band: string;
  outcome_class: string;
  snippet: string;
  quality_flags: string[];
}

export interface SearchResponse {
  query: string;
  filters: SearchFilters;
  results: CaseResult[];
  took_ms: number;
  embedding_version: string;
  reranked?: boolean;
}

export interface SimilarityFactor {
  factor: string;
  detail: string;
  citations: string[];
}

export interface Difference {
  detail: string;
  citations: string[];
}

export interface TreatmentObserved {
  treatment: string;
  outcome_note: string;
  citations: string[];
}

export interface CaseExplanation {
  case_id: string;
  similarity_factors: SimilarityFactor[];
  differences: Difference[];
  treatments_observed: TreatmentObserved[];
  confidence: "high" | "moderate" | "weak";
}

export interface ExplainResponse {
  query: string;
  explanations: CaseExplanation[];
  cohort_observation: string;
  disclaimer: string;
  model_used: string;
  degraded: boolean;
  cached: boolean;
  took_ms: number;
}

export interface CaseDetail {
  case_id: string;
  document: string;
  summary: Record<string, unknown> | null;
  sex: string;
  age: number | null;
  age_band: string;
  outcome_class: string;
  quality_flags: string[];
}

export interface HealthResponse {
  status: string;
  qdrant: boolean;
  redis: boolean;
  llm_configured: boolean;
  points_indexed: number;
}

const BASE = "/api/v1";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

export function search(
  query: string,
  k: number,
  filters: SearchFilters,
  rerank = false,
): Promise<SearchResponse> {
  return post<SearchResponse>("/search", { query, k, filters, rerank });
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

export function explain(
  query: string,
  caseIds: string[],
): Promise<ExplainResponse> {
  return post<ExplainResponse>("/explain", { query, case_ids: caseIds });
}

export async function getCase(caseId: string): Promise<CaseDetail> {
  const res = await fetch(`${BASE}/cases/${caseId}`);
  if (!res.ok) throw new Error(`case ${caseId} not found`);
  return res.json();
}
