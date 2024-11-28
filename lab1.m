load("10_fast_steps.mat")
temp = [[1: length(Acceleration.Timestamp)]', Acceleration{:, :}]
csvwrite("out.csv", temp)
