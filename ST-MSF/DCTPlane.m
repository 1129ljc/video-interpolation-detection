function Plane = DCTPlane(path)
  jobj=jpeg_read(path);
  Plane=jobj.coef_arrays{1};
end