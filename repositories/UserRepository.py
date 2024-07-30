class UserRepository:
    def __init__(self,):
        self.User = "User"

    def getAll(self):
        try:
            return self.User.query.all()
        except Exception as e:
            print("Error: "+ str(e))