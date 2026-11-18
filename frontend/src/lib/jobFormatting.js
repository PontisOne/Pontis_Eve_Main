function humanizeEmploymentType(value) {
  if (value == null) return "";
  const humanized = String(value)
    .replace(/[\[\]{}\"']/g, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (character) => character.toUpperCase());

  return humanized.replace(/^Full Time$/i, "Full-time").replace(/^Part Time$/i, "Part-time");
}

export function formatEmploymentType(value) {
  const values = Array.isArray(value) ? value : [value];
  const formatted = values
    .flatMap((item) => (Array.isArray(item) ? item : [item]))
    .map(humanizeEmploymentType)
    .filter(Boolean);

  return [...new Set(formatted)].join(", ");
}
