import re
import os
from difflib import SequenceMatcher

from python.helpers.tool import Tool, Response
from python.helpers.files import read_file, write_file


class LineReplace(Tool):
    async def execute(self, target_file: str = "", search: str = "", replace: str = "", **kwargs):
        """
        Replace text in a file using simple search and replace.
        
        Args:
            target_file: Absolute path to the file to modify
            search: Text to search for (exact match)
            replace: Text to replace with (empty string for deletion)
        """
        
        if not target_file:
            return Response(message="❌ Error: target_file parameter is required", break_loop=False)
            
        if not search:
            return Response(message="❌ Error: search parameter is required", break_loop=False)
            
        if not os.path.isabs(target_file):
            return Response(message="❌ Error: target_file must be an absolute path", break_loop=False)
            
        if not os.path.exists(target_file):
            return Response(message=f"❌ Error: File does not exist: {target_file}", break_loop=False)
            
        try:
            # Read original file
            original_content = read_file(target_file)
            
            # Apply replacement
            if search in original_content:
                # Exact match found
                modified_content = original_content.replace(search, replace, 1)
                
                # Write modified content back to file
                write_file(target_file, modified_content)
                
                # Create summary
                if replace.strip():
                    action = "replaced"
                    preview = f"'{search[:50]}...' → '{replace[:50]}...'" if len(search) > 50 or len(replace) > 50 else f"'{search}' → '{replace}'"
                else:
                    action = "deleted"
                    preview = f"'{search[:50]}...'" if len(search) > 50 else f"'{search}'"
                
                result_message = f"✅ File successfully modified: {target_file}\n\n"
                result_message += f"📋 Change applied: {action} {preview}"
                    
                return Response(message=result_message, break_loop=False)
            else:
                # Try fuzzy matching
                lines = original_content.split('\n')
                search_lines = search.split('\n')
                
                best_match = None
                best_ratio = 0.0
                
                # Try to find best fuzzy match
                for i in range(len(lines) - len(search_lines) + 1):
                    candidate_lines = lines[i:i + len(search_lines)]
                    candidate_text = '\n'.join(candidate_lines)
                    
                    ratio = SequenceMatcher(None, search, candidate_text).ratio()
                    
                    if ratio > best_ratio and ratio >= 0.8:  # High threshold for fuzzy matching
                        best_ratio = ratio
                        best_match = candidate_text
                
                if best_match:
                    # Apply fuzzy match
                    modified_content = original_content.replace(best_match, replace, 1)
                    write_file(target_file, modified_content)
                    
                    if replace.strip():
                        action = "replaced (fuzzy match)"
                        preview = f"'{search[:50]}...' → '{replace[:50]}...'" if len(search) > 50 or len(replace) > 50 else f"'{search}' → '{replace}'"
                    else:
                        action = "deleted (fuzzy match)"
                        preview = f"'{search[:50]}...'" if len(search) > 50 else f"'{search}'"
                    
                    result_message = f"✅ File successfully modified: {target_file}\n\n"
                    result_message += f"📋 Change applied: {action} {preview}\n"
                    result_message += f"📊 Match confidence: {best_ratio:.1%}"
                        
                    return Response(message=result_message, break_loop=False)
                else:
                    return Response(message=f"ℹ️ No changes applied to file: {target_file}\n\n❌ Search text not found (tried exact and fuzzy matching)", break_loop=False)
                
        except Exception as e:
            return Response(message=f"❌ Error processing file {target_file}: {str(e)}", break_loop=False)