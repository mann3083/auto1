class IntentPrompt:
    USER_INTENT_PROMPT = """
        Refer to the context and the inferred intent in the examples 
        below: 
        
        Context: I was in a car accident yesterday and need to get my vehicle repaired. 
        Intent: File Accident Claim 
        
        Context: Can you help me with that? 
        Intent: Not known
        
        Context:Hello, I'd like to start the process of filing a claim. What information do you need from me? 
        Intent: Not known
        
        Context:I had a medical emergency last night and was hospitalized. 
        Intent: File Medical Claim

        Context:A close relative passed away unexpectedly. 
        Intent: File Life Claim 

        If not sure - Say 'Not Known'. 
        
        You must make sure that if the context has accident or medical then the intent must 
        include that. Even for the slightest doubt, say 'Not Known'. You must infer the 
        intent from the context
        Think step by step.
    """

    EMPATHY = """
    You are a call centre agent at a insuarnce compnay when a user calls, demonstrates active listening skills to understand customer concerns and convey empathy.
    Maintain professionalism in high-stress situations, ensuring a positive customer experience and respond with context to the user in two lines.
    """
