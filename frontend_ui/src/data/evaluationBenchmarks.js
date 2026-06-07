/**
 * Offline evaluation metrics (held-out test sets). Replace with values from
 * scripts/compare_four_models_metrics.py or your dissertation tables.
 * Not fetched from the API — edit here when you freeze numbers.
 */
export const MODEL_EVALUATION_ROWS = [
  {
    key: 'tfidf-fakeddit',
    label: 'TF–IDF + LR',
    corpus: 'Social-style test',
    accuracy: 0.847,
    precision: 0.831,
    recall: 0.819,
    f1: 0.825,
  },
  {
    key: 'distil-fakeddit',
    label: 'Encoder (quick)',
    corpus: 'Social-style test',
    accuracy: 0.891,
    precision: 0.884,
    recall: 0.862,
    f1: 0.873,
  },
  {
    key: 'hybrid-fakeddit',
    label: 'Full fusion',
    corpus: 'Social-style test',
    accuracy: 0.912,
    precision: 0.901,
    recall: 0.888,
    f1: 0.894,
  },
  {
    key: 'distil-liar',
    label: 'Encoder (quick)',
    corpus: 'News-style test',
    accuracy: 0.623,
    precision: 0.601,
    recall: 0.587,
    f1: 0.594,
  },
  {
    key: 'hybrid-liar',
    label: 'Full fusion',
    corpus: 'News-style test',
    accuracy: 0.641,
    precision: 0.618,
    recall: 0.605,
    f1: 0.611,
  },
];

/** Confusion matrix for the strongest model on the social-style test (example). */
export const CONFUSION_BEST_MODEL = {
  title: 'Full fusion — social-style test',
  labels: ['Reliable (pred.)', 'Unreliable (pred.)'],
  matrix: [
    [4820, 410],
    [360, 4410],
  ],
  note: 'Rows: true label. Columns: predicted label. Replace with your exported matrix.',
};

/** Optional user-study results (paste after running a questionnaire). */
export const QUESTIONNAIRE_SAMPLE = {
  n: 24,
  clarity: 4.2,
  usefulness: 4.1,
  trust: 3.7,
  comments:
    'Illustrative averages on a 1–5 Likert scale. Connect a real form or CSV import when available.',
};
