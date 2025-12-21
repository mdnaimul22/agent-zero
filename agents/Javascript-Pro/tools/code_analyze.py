import os
import json
from typing import Dict, Any, List

from python.helpers.tool import Tool, Response
from python.helpers.print_style import PrintStyle
from python.helpers.errors import handle_error
from python.helpers.code_analyzer import LanguageDetector, PythonAnalyzer, JavaScriptAnalyzer, CodeMetricsCalculator

class CodeAnalyze(Tool):
    async def execute(self, **kwargs) -> Response:
        file_path = kwargs.get("file_path")
        analysis_type = kwargs.get("analysis_type", "all")
        
        if not file_path:
            return Response(message="Error: file_path parameter is required", break_loop=False)
        
        if not os.path.isfile(file_path):
            return Response(message=f"Error: File {file_path} does not exist", break_loop=False)
        
        try:
            language_detector = LanguageDetector()
            metrics_calculator = CodeMetricsCalculator()
            
            language = language_detector.detect_language(file_path)
            metrics = metrics_calculator.calculate_metrics(file_path)
            
            analysis_results = {
                "file": file_path,
                "language": language,
                "metrics": metrics
            }
            
            if analysis_type in ["all", "complexity", "structure"]:
                if language == "python":
                    analyzer = PythonAnalyzer()
                    complexity_results = analyzer.analyze_complexity(file_path)
                    analysis_results["structure"] = complexity_results
                elif language in ["javascript", "typescript"]:
                    analyzer = JavaScriptAnalyzer()
                    complexity_results = analyzer.analyze_complexity(file_path)
                    analysis_results["structure"] = complexity_results
            
            if analysis_type in ["all", "style"] and language == "python":
                analyzer = PythonAnalyzer()
                style_results = analyzer.check_style(file_path)
                analysis_results["style"] = style_results
            
            # Format the response as markdown
            response = self._format_results(analysis_results)
            
            # Log the action
            self.log.update(content=f"Analyzed {file_path}")
            
            return Response(message=response, break_loop=False)
            
        except Exception as e:
            handle_error(e)
            return Response(message=f"Error analyzing file: {str(e)}", break_loop=False)
    
    def _format_results(self, results: Dict[str, Any]) -> str:
        response = f"# Code Analysis for {results['file']}\n\n"
        
        response += f"## General Information\n"
        response += f"- **Language**: {results['language']}\n"
        
        if "metrics" in results:
            metrics = results["metrics"]
            response += f"- **Total Lines**: {metrics.get('total_lines', 'N/A')}\n"
            response += f"- **Code Lines**: {metrics.get('code_lines', 'N/A')}\n"
            response += f"- **Comment Lines**: {metrics.get('comment_lines', 'N/A')}\n"
            response += f"- **Blank Lines**: {metrics.get('blank_lines', 'N/A')}\n"
            response += f"- **Comment Ratio**: {metrics.get('comment_ratio', 'N/A')}%\n\n"
        
        if "structure" in results:
            structure = results["structure"]
            response += f"## Code Structure\n"
            
            if "imports" in structure and structure["imports"]:
                response += f"### Imports ({len(structure['imports'])})\n"
                for imp in structure["imports"][:10]:  # Limit to first 10 imports
                    response += f"- `{imp}`\n"
                if len(structure["imports"]) > 10:
                    response += f"- ... and {len(structure['imports']) - 10} more\n"
                response += "\n"
            
            if "classes" in structure and structure["classes"]:
                response += f"### Classes ({len(structure['classes'])})\n"
                for cls in structure["classes"]:
                    response += f"- **{cls['name']}** (line {cls.get('line', 'N/A')})\n"
                    if "methods" in cls and cls["methods"]:
                        for method in cls["methods"][:5]:  # Limit to first 5 methods
                            response += f"  - `{method}`\n"
                        if len(cls["methods"]) > 5:
                            response += f"  - ... and {len(cls['methods']) - 5} more methods\n"
                response += "\n"
            
            if "functions" in structure and structure["functions"]:
                response += f"### Functions ({len(structure['functions'])})\n"
                for func in structure["functions"][:10]:  # Limit to first 10 functions
                    args_str = ", ".join(func.get("args", []))
                    response += f"- **{func['name']}**({args_str}) (line {func.get('line', 'N/A')})\n"
                if len(structure["functions"]) > 10:
                    response += f"- ... and {len(structure['functions']) - 10} more functions\n"
                response += "\n"
        
        if "style" in results and "style_issues" in results["style"] and results["style"]["style_issues"]:
            issues = results["style"]["style_issues"]
            response += f"## Style Issues ({len(issues)})\n"
            for issue in issues[:10]:  # Limit to first 10 issues
                response += f"- {issue}\n"
            if len(issues) > 10:
                response += f"- ... and {len(issues) - 10} more issues\n"
        
        return response
