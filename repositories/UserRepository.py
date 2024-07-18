class UserRepository:
    def __init__(self,):
        self.User = "User"

    def getAll(self):
        try:
            return "test:prvni"
        except Exception as e:
            print("Error: "+ str(e))