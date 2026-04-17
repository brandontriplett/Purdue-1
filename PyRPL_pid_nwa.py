from pyrpl import Pyrpl

p = Pyrpl()
r = p.rp

# Configure PID
pid = r.pid0

pid.input = 'in1'
pid.output = 'out1'

pid.p = 0.3
pid.i = 10
pid.d = 0

pid.enable()

# Network analyzer
na = p.networkanalyzer

na.input = 'out1'
na.output_direct = 'out1'

na.start_freq = 10
na.stop_freq = 1e6
na.points = 200

data = na.single()