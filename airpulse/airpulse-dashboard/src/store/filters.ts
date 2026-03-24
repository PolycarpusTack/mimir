import { create } from 'zustand';
import type { Domain, SignalType } from '@/types';

export type SortOrder = 'relevance' | 'recent';

interface FilterStore {
  filterDomain: Domain | null;
  filterSignalType: SignalType | null;
  sortOrder: SortOrder;
  setFilterDomain: (d: Domain | null) => void;
  setSignalType: (t: SignalType | null) => void;
  setSortOrder: (o: SortOrder) => void;
}

export const useFilterStore = create<FilterStore>((set) => ({
  filterDomain: null,
  filterSignalType: null,
  sortOrder: 'relevance',
  setFilterDomain: (d) => set({ filterDomain: d }),
  setSignalType: (t) => set({ filterSignalType: t }),
  setSortOrder: (o) => set({ sortOrder: o }),
}));
