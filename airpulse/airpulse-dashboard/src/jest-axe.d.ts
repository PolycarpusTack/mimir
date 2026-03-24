declare module 'jest-axe' {
  import type { AxeResults } from 'axe-core';
  export function axe(container: Element | Document, options?: Record<string, unknown>): Promise<AxeResults>;
  export function toHaveNoViolations(): {
    message(): string;
    pass: boolean;
  };
}
