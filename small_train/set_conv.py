import set_vars as sv

def set_conv(sess):
  # Force conv1 to see features exactly. Diamond, cross, hor, ver.
  sv.set_conv1(sess)

  # Set conv2 too.
  sv.set_conv2(sess)
