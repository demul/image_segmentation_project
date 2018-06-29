import FCN

batch_size = 8
max_epoch = 5000
learnig_rate = 0.0001
already_done_epoch = 27

model = FCN.FCN(batch_size, learnig_rate)
model.run(max_epoch, already_done_epoch)


