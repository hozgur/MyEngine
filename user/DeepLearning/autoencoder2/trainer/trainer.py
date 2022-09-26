

def log(msg):
    my.Message(json.dumps({"id": "python", "message":msg}))


des = np.asarray(Image.open(my.Path("user/python/graphics/test.png")))
log("image loaded.")
log(str(des.shape))
