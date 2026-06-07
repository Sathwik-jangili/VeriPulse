/**
 * Recharts Tooltip defaults ignore text colour; without this, labels render black on dark UI.
 */
export function getChartTooltipProps(darkMode = true) {
  const dark = darkMode !== false;
  return {
    contentStyle: {
      backgroundColor: dark ? '#09090b' : '#ffffff',
      border: dark ? '1px solid #52525b' : '1px solid #d4d4d8',
      borderRadius: '8px',
      fontSize: '12px',
      color: dark ? '#fafafa' : '#18181b',
      boxShadow: dark ? '0 10px 15px -3px rgb(0 0 0 / 0.45)' : '0 4px 6px -1px rgb(0 0 0 / 0.1)',
    },
    labelStyle: {
      color: dark ? '#f4f4f5' : '#18181b',
      fontWeight: 600,
      marginBottom: '4px',
    },
    itemStyle: {
      color: dark ? '#e4e4e7' : '#3f3f46',
    },
    wrapperStyle: { outline: 'none' },
  };
}
