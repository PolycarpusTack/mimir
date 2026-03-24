export interface VendorEntry {
  name: string;
  tier: 'competitor' | 'partner' | 'adjacent';
}

export const VENDOR_WATCHLIST: VendorEntry[] = [
  { name: 'Imagine Communications', tier: 'competitor' },
  { name: 'Harmonic', tier: 'competitor' },
  { name: 'Evertz', tier: 'competitor' },
  { name: 'Grass Valley', tier: 'competitor' },
  { name: 'Vizrt', tier: 'competitor' },
  { name: 'Dalet', tier: 'competitor' },
  { name: 'Avid', tier: 'competitor' },
  { name: 'Pebble', tier: 'competitor' },
  { name: 'BroadPeak', tier: 'competitor' },
  { name: 'Ateme', tier: 'competitor' },
  { name: 'AWS Elemental', tier: 'partner' },
  { name: 'Microsoft Azure', tier: 'partner' },
  { name: 'Google Cloud', tier: 'partner' },
  { name: 'Akamai', tier: 'partner' },
  { name: 'Fastly', tier: 'partner' },
  { name: 'Comcast', tier: 'adjacent' },
  { name: 'Disney', tier: 'adjacent' },
  { name: 'Netflix', tier: 'adjacent' },
  { name: 'Warner Bros Discovery', tier: 'adjacent' },
  { name: 'Paramount', tier: 'adjacent' },
  { name: 'NBCUniversal', tier: 'adjacent' },
  { name: 'Fox', tier: 'adjacent' },
  { name: 'BBC', tier: 'adjacent' },
  { name: 'ITV', tier: 'adjacent' },
  { name: 'RTL', tier: 'adjacent' },
  { name: 'ProSieben', tier: 'adjacent' },
  { name: 'TF1', tier: 'adjacent' },
  { name: 'Roku', tier: 'adjacent' },
  { name: 'Brightcove', tier: 'adjacent' },
  { name: 'Verizon', tier: 'adjacent' },
];
