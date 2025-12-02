export interface Book {
  title: string;
  he_title?: string;
  categories: string[];
  path: string;
}

export interface Category {
  name: string;
  he_name?: string;
  books: Book[];
  subcategories: Category[];
}

export interface BooksResponse {
  categories: Category[];
  total_books: number;
}

export interface SearchRequest {
  query: string;
  limit?: number;
  score_threshold?: number;
  book?: string;
  category?: string[];
  embedding_model?: string;
}

export interface SearchResult {
  sefaria_ref: string;
  book: string;
  category: string[];
  text: string;
  score: number;
  position: number;
  chunk_type: string;
  masechet?: string;
  daf?: number;
  amud?: string;
  chapter?: number;
  verse?: number;
  perek?: number;
  mishnah?: number;
  part?: string;
  siman?: number;
  seif?: number;
  embedding_model?: string;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  query: string;
}

export interface PipelineRequest {
  books?: string[];
  embedding_model?: string;
  language?: string;
}

export interface PipelineResponse {
  message: string;
  books?: string[];
  status: string;
}

export interface DeleteResponse {
  message: string;
  deleted_count: number;
  book?: string | null;
}
