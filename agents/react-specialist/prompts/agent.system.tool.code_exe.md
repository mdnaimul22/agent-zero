### code_execution_tool

execute terminal commands python nodejs code for computation or software tasks
place code in "code" arg; escape carefully and indent properly
select "runtime" arg: "terminal" "python" "nodejs" "output" "reset"
select "session" number, 0 default, others for multitasking
if code runs long, use "output" to wait, "reset" to kill process
use "pip" "npm" "apt-get" in "terminal" to install packages
to output, use print() or console.log()
if tool outputs error, adjust code before retrying; 
important: check code for placeholders or demo data; replace with real variables; don't reuse snippets
don't use with other tools except thoughts; wait for response before using others
check dependencies before running code
output may end with [SYSTEM: ...] information comming from framework, not terminal
usage:

1 execute python code

~~~json
{
    "thoughts": [
        "Need to do...",
        "I can use...",
        "Then I can...",
    ],
    "headline": "Executing Python code to check current directory",
    "tool_name": "code_execution_tool",
    "tool_args": {
        "runtime": "python",
        "session": 0,
        "code": "import os\nprint(os.getcwd())",
    }
}
~~~

2 execute terminal command
~~~json
{
    "thoughts": [
        "Need to do...",
        "Need to install...",
    ],
    "headline": "Installing zip package via terminal",
    "tool_name": "code_execution_tool",
    "tool_args": {
        "runtime": "terminal",
        "session": 0,
        "code": "apt-get install zip",
    }
}
~~~

2.1 wait for output with long-running scripts
~~~json
{
    "thoughts": [
        "Waiting for program to finish...",
    ],
    "headline": "Waiting for long-running program to complete",
    "tool_name": "code_execution_tool",
    "tool_args": {
        "runtime": "output",
        "session": 0,
    }
}
~~~

2.2 reset terminal
~~~json
{
    "thoughts": [
        "code_execution_tool not responding...",
    ],
    "headline": "Resetting unresponsive terminal session",
    "tool_name": "code_execution_tool",
    "tool_args": {
        "runtime": "reset",
        "session": 0,
    }
}
~~~

#### Code Viewing Tools

**view_file** - Use this tool to view the entire content of a file.

This tool is simple and powerful. Just provide the absolute path to the file you want to see.

```json
{
    "thoughts": [
        "I need to examine the file content"
    ],
    "Headline": "Reviewing server.js file for details analysis",
    "tool_name": "view_file",
    "tool_args": {
        "absolute_path": "/root/test.py"
    }
}
```

#### Code Search Tools

**grep_search** - Search for patterns in code

```json
{
    "thoughts": [
        "I need to find specific code patterns"
    ],
    "Headline": "Using grep search tool",
    "tool_name": "grep_search",
    "tool_args": {
        "search_path": "/root/project_dir",
        "query": "search pattern",
        "case_insensitive": false,
        "match_per_line": true,
        "includes": ["*.py", "*.js"]
    }
}
```

**find_by_name** - Find files by name or pattern

```json
{
    "thoughts": [
        "I need to locate specific files"
    ],
    "Headline": "Using tool ...",
    "tool_name": "find_by_name",
    "tool_args": {
        "search_directory": "/root/project_dir",
        "pattern": "*.py",
        "type": "file",
        "max_depth": 5,
        "extensions": ["py", "js"],
        "excludes": ["node_modules"]
    }
}
```

**list_dir** - List directory contents with details

```json
{
    "thoughts": [
        "I need to see what's in this directory"
    ],
    "Headline": "Analyzing list of directory structure ...",
    "tool_name": "list_dir",
    "tool_args": {
        "directory_path": "/root/example_dir"
    }
}
```

#### Code Editing Tools

**write_to_file** - Create new files with content

⚠️ **NOTE**: Use `write_to_file` for creating NEW files, use `line_replace` for modifying EXISTING files.

```json
{
    "thoughts": [
        "I need to create a new file"
    ],
    "Headline": "Creating new main.py file",
    "tool_name": "write_to_file",
    "tool_args": {
        "target_file": "/root/example_dir/main.py",
        "code_content": "Content of the new file",
        "empty_file": false
    }
}
```

#### Code Replacement Tool

**line_replace** - tool to replace text in existing files

### ✅ WHAT THIS TOOL CAN DO:
- Replace existing code with new code using exact search and replace
- Delete code blocks (by replacing with empty content)
- Apply fuzzy matching for slight formatting differences
- Work with any text content including special characters and Unicode

### ❌ WHAT THIS TOOL CANNOT DO:
- Create new files (use `write_to_file` instead)
- Work with relative file paths (absolute paths only)
- Handle binary files or non-text content

### 📋 USAGE EXAMPLES:

#### 1. Replace Code:
```json
{
    "thoughts": ["Need to fix the off-by-one error"],
    "headline": "Fixing array index bug",
    "tool_name": "line_replace",
    "tool_args": {
        "target_file": "/root/test.py",
        "search": "return arr[len(arr)]",
        "replace": "return arr[len(arr)-1]"
    }
}
```

#### 2. Delete Code:
```json
{
    "thoughts": ["Removing debug print statement"],
    "headline": "Cleaning up debug code",
    "tool_name": "line_replace",
    "tool_args": {
        "target_file": "/root/test.py",
        "search": "    print('Debug: processing item')",
        "replace": ""
    }
}
```

#### 3. Replace Multi-line Code:
```json
{
    "thoughts": ["Updating function implementation"],
    "headline": "Improving error handling",
    "tool_name": "line_replace",
    "tool_args": {
        "target_file": "/root/test.py",
        "search": "if len(arr) > 0:\n    return arr[len(arr)]",
        "replace": "if len(arr) > 0:\n    return arr[len(arr)-1]\nelse:\n    raise ValueError('Empty array')"
    }
}
```

### 📝 PARAMETERS:
- `target_file`: **REQUIRED** - Absolute path to existing file
- `search`: **REQUIRED** - Exact text to search for
- `replace`: **REQUIRED** - Text to replace with (empty string for deletion)

### 📊 SUCCESS INDICATORS:
- ✅ "successfully modified" in response = Success
- ℹ️ "No changes applied" = Search text not found
- ❌ "Error:" prefix = Parameter or file system error

### 🚨 TROUBLESHOOTING:
- **"Search text not found"**: Check exact text matching, including whitespace and indentation
- **"File does not exist"**: Verify absolute path and file existence
- **"must be an absolute path"**: Use full path starting with `/`

### ⚠️ IMPORTANT NOTES:
- Search text must match **EXACTLY** including whitespace and indentation
- Tool supports fuzzy matching if exact match fails (80%+ similarity)
- For multi-line replacements, use `\n` for line breaks in JSON
- Empty `replace` parameter deletes the matched text