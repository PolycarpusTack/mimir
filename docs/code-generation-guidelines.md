### Code Generation Guidelines for AI:

1. **Import Path Verification**:
   - Always verify the actual file structure before generating imports
   - Use the exact directory names (check for singular vs plural)
   - Ensure relative import paths are correct based on file location

2. **Type Safety**:
   - Initialize all class properties or mark them as optional with `?`
   - For DTOs, use class initialization syntax:
     ```typescript
     @IsString()
     name!: string;  // Use definite assignment assertion
     ```

3. **Module Dependencies**:
   - List all cross-module dependencies explicitly
   - Ensure circular dependencies are avoided
   - Use forwardRef() when necessary for circular dependencies

4. **Consistent Naming**:
   - Follow consistent naming patterns (singular for files, plural for directories)
   - Match import names with actual file/directory names
   - Use index.ts files for cleaner imports

5. **Validation Before Generation**:
   - Check if referenced modules/files exist
   - Verify TypeScript strict mode settings
   - Ensure all required dependencies are installed

6. **Error Prevention**:
   - Add explicit type annotations
   - Avoid implicit any types
   - Handle all possible null/undefined cases
   - Use proper error types instead of generic Error

7. **Build Testing**:
   - Generate code that compiles without errors
   - Include all necessary imports
   - Ensure no unused imports or variables