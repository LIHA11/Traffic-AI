class InternalServerError(Exception):
    message: str
    code: int = 500

    def __init__(self, message="Internal Server Error"):
        self.message = message
        super().__init__(self.message)