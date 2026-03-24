import { describe, it, expect, afterEach } from 'vitest';
import { useUIStore } from '@/store/ui';
import { useFilterStore } from '@/store/filters';

afterEach(() => {
  useUIStore.setState({
    selectedSignalId: null,
    activePanel: 'shift',
    queueExpanded: false,
  });
  useFilterStore.setState({
    filterDomain: null,
    filterSignalType: null,
    sortOrder: 'relevance',
  });
});

describe('UIStore', () => {
  it('sets selected signal ID', () => {
    useUIStore.getState().setSelectedSignalId('sig-001');
    expect(useUIStore.getState().selectedSignalId).toBe('sig-001');
  });

  it('sets active panel', () => {
    useUIStore.getState().setActivePanel('vendors');
    expect(useUIStore.getState().activePanel).toBe('vendors');
  });

  it('toggles queue', () => {
    expect(useUIStore.getState().queueExpanded).toBe(false);
    useUIStore.getState().toggleQueue();
    expect(useUIStore.getState().queueExpanded).toBe(true);
    useUIStore.getState().toggleQueue();
    expect(useUIStore.getState().queueExpanded).toBe(false);
  });

  it('clears selected signal', () => {
    useUIStore.getState().setSelectedSignalId('sig-001');
    useUIStore.getState().setSelectedSignalId(null);
    expect(useUIStore.getState().selectedSignalId).toBeNull();
  });
});

describe('FilterStore', () => {
  it('sets domain filter', () => {
    useFilterStore.getState().setFilterDomain('AI');
    expect(useFilterStore.getState().filterDomain).toBe('AI');
  });

  it('clears domain filter', () => {
    useFilterStore.getState().setFilterDomain('AI');
    useFilterStore.getState().setFilterDomain(null);
    expect(useFilterStore.getState().filterDomain).toBeNull();
  });

  it('sets sort order', () => {
    useFilterStore.getState().setSortOrder('recent');
    expect(useFilterStore.getState().sortOrder).toBe('recent');
  });

  it('sets signal type filter', () => {
    useFilterStore.getState().setSignalType('MaSignal');
    expect(useFilterStore.getState().filterSignalType).toBe('MaSignal');
  });
});
