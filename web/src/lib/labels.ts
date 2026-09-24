// Plain-language names for the policy's gates and flags. The study page and the portfolio both
// show these; the raw names stay in the evidence, the brief and policy.yaml.

export const GATE_TEXT: Record<string, string> = {
  data: "Enough of the fleet emits every needed signal",
  model: "The model beats the best single signal",
  economics: "A positive return is more likely than not",
  cost: "Run cost is under the requested ceiling",
  delivery: "A delivery pattern fits the horizon",
};

export const FLAG_TEXT: Record<string, string> = {
  cross_oem_variance: "Performance differs across OEMs",
  temporal_degradation: "Performance decays over time",
  roi_spans_negative: "The return could be negative in a plausible case",
  value_unvalidated: "Value assumptions have not been validated by a pilot",
  short_history: "A needed signal has too little history",
  ablation_underpowered: "A signal removal rests on a point estimate",
  suspicious_signals: "A needed signal is unusually strong; confirm it exists before the event, not because of it",
  cost_placeholders: "Run cost uses placeholder prices",
  alert_burden: "Too many false alerts at the chosen operating point",
  data_still_improving: "More data would still raise the accuracy",
  seed_sensitive: "The result depends on how vehicles were split",
};

// short forms for dense tables
export const GATE_SHORT: Record<string, string> = {
  data: "fleet coverage too low",
  model: "no better than one signal",
  economics: "return not likely positive",
  cost: "over the cost ceiling",
  delivery: "cannot be delivered in time",
};

export const FLAG_SHORT: Record<string, string> = {
  cross_oem_variance: "varies by OEM",
  temporal_degradation: "decays over time",
  roi_spans_negative: "return could be negative",
  value_unvalidated: "value not validated",
  short_history: "short signal history",
  ablation_underpowered: "underpowered ablation",
  suspicious_signals: "signal needs confirming",
  cost_placeholders: "placeholder prices",
  alert_burden: "too many false alerts",
  data_still_improving: "more data would help",
  seed_sensitive: "depends on the split",
};
