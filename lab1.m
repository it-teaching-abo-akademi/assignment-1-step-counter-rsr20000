load("10_run.mat")
temp = [[1: length(Acceleration.Timestamp)]', Acceleration{:, :}]
csvwrite("out.csv", temp)