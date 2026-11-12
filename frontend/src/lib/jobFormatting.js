function humanizeEmploymentType(value) {
  return String(value)
    .replace(/[\[\]{}\"']/g, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export function formatEmploymentType(value) {
  const values = Array.isArray(value) ? value : [value];
  const formatted = values
    .flatMap((item) => (Array.isArray(item) ? item : [item]))
    .map(humanizeEmploymentType)
    .filter(Boolean);

  return [...new Set(formatted)].join(", ");
}
