few_shots = [
    {
        "Question": "How many users do we have?",
        "SQLQuery": "SELECT COUNT(*) FROM users",
        "SQLResult": "Result of the SQL query",
        "Answer": "1118"
    },
    {
        "Question": "How many facilities or clinics do we have?",
        "SQLQuery": "SELECT COUNT(*) FROM facility",
        "SQLResult": "Result of the SQL query",
        "Answer": "62"
    },
    {
        "Question": "How many users are in Amanora clinics?",
        "SQLQuery": """
            SELECT COUNT(*) 
            FROM users  
            INNER JOIN facility ON users.facility_id = facility.id 
            WHERE facility.name = 'Amanora'
        """,
        "SQLResult": "Result of the SQL query",
        "Answer": "5"
    },
    {
        "Question": "How many receipts do we have for last year?",
        "SQLQuery": """
            SELECT COUNT(*) 
            FROM reciept 
            WHERE YEAR(rect_created_date) = YEAR(CURDATE()) - 1
        """,
        "SQLResult": "Result of the SQL query",
        "Answer": "480"
    },
    {
        "Question": "What is the cost of 'RCT with Rubber Dam - By consultant, using rotary files with endomotor, apex locator, permanent fill' treatment?",
        "SQLQuery": """
            SELECT trname, SUM(trprice) AS total_price 
            FROM treatment_master 
            WHERE trname LIKE '%RCT with Rubber Dam%' 
            GROUP BY trname
        """,
        "SQLResult": "Result of the SQL query",
        "Answer": "5000"
    },
    {
        "Question": "what is pubpid for patient deepak_17_pay_reco_to ?",
        "SQLQuery": """
            SELECT pubpid FROM patient_data WHERE fname = 'deepak_17_pay_reco_to' LIMIT 1;
        """,
        "SQLResult": "Result of the SQL query",
        "Answer": "5000"
    },
    {
        "Question": "what is pubpid for patient deepak_17_pay_reco_to ?",
        "SQLQuery": """
            SELECT pubpid FROM patient_data WHERE fname = 'deepak_17_pay_reco_to' LIMIT 1;
        """,
        "SQLResult": "Result of the SQL query",
        "Answer": "5000"
    },
    {
        "Question": "what is name and email for patient deepak_17_pay_reco_to ?",
        "SQLQuery": """
            SELECT pubpid FROM patient_data WHERE fname = 'deepak_17_pay_reco_to' LIMIT 1;
        """,
        "SQLResult": "Result of the SQL query",
        "Answer": "5000"
    }

]