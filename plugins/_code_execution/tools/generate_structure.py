from pathlib import Path
from helpers.tool import Tool, Response


class GenerateStructure(Tool):
    """
    Tool to generate a directory structure in Markdown format.
    
    Args:
        tree_structure_input_path: Directory to scan (required)
        tree_structure_out_path: Directory where structure.md will be written (defaults to input path)
        ignored_path: Comma-separated list of additional paths to ignore
        ignored_extensions: Comma-separated list of patterns/extensions to ignore
    """

    # Default ignored patterns (common build artifacts, dependencies, and media files)
    DEFAULT_IGNORED_PATTERNS = {
        # Version control and IDE
        ".git",
        ".gitkeep",
        ".idea",
        ".vscode",
        ".DS_Store",
        
        # Python
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        ".egg-info",
        ".pyc",
        ".pyo",
        ".pyd",
        
        # Node.js
        "node_modules",
        ".npm",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        
        # Build outputs
        "dist",
        "build",
        "coverage",
        
        # Lock files
        "poetry.lock",
        "Gemfile.lock",
        ".lock",
        
        # Environment files
        ".env",
        ".venv",
        
        # Media files
        ".svg",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tiff",
        ".ico",
        ".mp4",
        ".mp3",
        
        # Web assets
        ".html",
        ".css",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        
        # Binaries and archives
        ".so",
        ".dll",
        ".exe",
        ".bin",
        ".iso",
        ".tar",
        ".gz",
        ".zip",
        
        # Temporary and log files
        ".log",
        ".tmp",
        ".bak",
        ".swp",
        ".swo",
        
        # Other
        ".java",
    }

    async def execute(self, **kwargs):
        # Get arguments
        input_path_str = self.args.get("tree_structure_input_path", "")
        output_path_str = self.args.get("tree_structure_out_path", input_path_str)
        ignored_path_str = self.args.get("ignored_path", "")
        ignored_extensions_str = self.args.get("ignored_extensions", "")

        # Validate input path
        if not input_path_str:
            return Response(
                message="Error: tree_structure_input_path is required",
                break_loop=False
            )

        # Resolve paths
        input_path = Path(input_path_str).resolve()
        output_path = Path(output_path_str).resolve()

        if not input_path.exists() or not input_path.is_dir():
            return Response(
                message=f"Error: Directory '{input_path_str}' does not exist or is not a directory",
                break_loop=False
            )

        # Parse ignored paths and extensions
        ignored_paths = set()
        if ignored_path_str:
            for path_str in ignored_path_str.split(","):
                path_str = path_str.strip()
                if path_str:
                    try:
                        resolved = Path(path_str).resolve()
                        ignored_paths.add(str(resolved))
                    except Exception:
                        # If resolution fails, add as-is
                        ignored_paths.add(path_str)

        # Combine default and custom ignored patterns
        ignored_patterns = set(self.DEFAULT_IGNORED_PATTERNS)
        if ignored_extensions_str:
            for pattern in ignored_extensions_str.split(","):
                pattern = pattern.strip()
                if pattern:
                    ignored_patterns.add(pattern)

        # Generate structure
        output_file_path = output_path / "structure.md"
        
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(f"# Project Structure: {input_path.name}\n")
                f.write(f"Root: `{input_path}`\n\n")
                
                self._generate_structure(
                    input_path,
                    f,
                    indent_level=0,
                    target_root=input_path,
                    excluded_paths=ignored_paths,
                    ignored_patterns=ignored_patterns
                )
            
            return Response(
                message=(
                    f"✓ Directory structure generated successfully at: {output_file_path}\n"
                    f"Action Required: You MUST now read this structure file to understand the project architecture, "
                    f"identify key files/documentation, and summarize your observations to the user."
                ),
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error generating structure: {str(e)}",
                break_loop=False
            )

    def _generate_structure(
        self,
        current_path: Path,
        output_file,
        indent_level: int = 0,
        target_root: Path = None,
        excluded_paths: set = None,
        ignored_patterns: set = None
    ):
        """Recursively generate the directory structure"""
        if excluded_paths is None:
            excluded_paths = set()
        if ignored_patterns is None:
            ignored_patterns = set()

        try:
            # Sort items: directories first, then files, both alphabetically
            items = sorted(
                current_path.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except PermissionError:
            return

        for item in items:
            # Resolve absolute path for exclusion checking
            try:
                item_abs_path = str(item.resolve())
            except OSError:
                item_abs_path = str(item.absolute())

            # Check for skipping
            if item.name in ignored_patterns:
                continue

            # Check if the item ends with any of the ignored extensions
            if any(item.name.endswith(pattern) for pattern in ignored_patterns if pattern.startswith('.')):
                continue

            # Skip structure.md in target root
            if item.name == "structure.md" and item.parent == target_root:
                continue

            # Check against custom excluded paths
            if item_abs_path in excluded_paths:
                continue

            # Print item cleanly without emojis or excess formatting
            indent = "  " * indent_level
            if item.is_dir():
                output_file.write(f"{indent}- {item.name}/\n")
                self._generate_structure(
                    item,
                    output_file,
                    indent_level + 1,
                    target_root,
                    excluded_paths,
                    ignored_patterns
                )
            else:
                output_file.write(f"{indent}- {item.name}\n")
