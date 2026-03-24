export interface ZThresholdConfig {
  label: string;
  minZ: number;
  bg: string;
  border: string;
  text: string;
}

export const Z_THRESHOLD_COLORS: ZThresholdConfig[] = [
  { label: 'SURGE',    minZ: 3.5, bg: 'rgba(255,96,96,0.35)',   border: 'rgba(255,96,96,0.7)',   text: '#ff6060' },
  { label: 'SPIKE',    minZ: 2.5, bg: 'rgba(255,160,50,0.30)',  border: 'rgba(255,160,50,0.7)',  text: '#ffa032' },
  { label: 'ELEVATED', minZ: 1.5, bg: 'rgba(255,201,93,0.20)',  border: 'rgba(255,201,93,0.6)',  text: '#ffc95d' },
  { label: 'ABOVE',    minZ: 0.8, bg: 'rgba(93,222,157,0.12)',  border: 'rgba(93,222,157,0.4)',  text: '#5dde9f' },
  { label: 'NORMAL',   minZ: 0.0, bg: 'rgba(255,255,255,0.04)', border: 'rgba(255,255,255,0.1)', text: '#666' },
];
