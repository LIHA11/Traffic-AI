class SessionExpiredError(Exception):
    message: str
    code: int = 401
    
    def __init__(self, message="Session expired"):
        self.message = message
        super().__init__(self.message)