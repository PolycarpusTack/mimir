import { create } from 'zustand';

export type ActivePanel = 'shift' | 'vendors' | 'tech' | 'digest';

interface UIStore {
  selectedSignalId: string | null;
  activePanel: ActivePanel;
  queueExpanded: boolean;
  setSelectedSignalId: (id: string | null) => void;
  setActivePanel: (p: ActivePanel) => void;
  toggleQueue: () => void;
}

export const useUIStore = create<UIStore>((set) => ({
  selectedSignalId: null,
  activePanel: 'shift',
  queueExpanded: false,
  setSelectedSignalId: (id) => set({ selectedSignalId: id }),
  setActivePanel: (p) => set({ activePanel: p }),
  toggleQueue: () => set((s) => ({ queueExpanded: !s.queueExpanded })),
}));
