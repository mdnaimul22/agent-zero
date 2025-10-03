# Problem
 address already in use
# Solution
 
    1.  **Identify and kill the process:** Use the following command to find and kill the process using port number.
        ```bash
        netstat -tulnp | grep :0000 && kill $(netstat -tulnp | grep :0000 | awk '{print $7}' | cut -d'/' -f1)
        ```
    2.  **Restart the server:** Restart the server in the terminal.
    