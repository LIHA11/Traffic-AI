class UnauthorizedError(Exception):
    message: str
    code: int = 401
    
    def __init__(self, message="Unauthorized"):
        self.message = message
        super().__init__(self.message)