{ pkgs ? import <nixpkgs> {} }:

let stdenv = pkgs.stdenv;
    optional = stdenv.lib.optional;
    # N.B. Jupyter and TensorFlow can only coexist on Python 3.6, at
    # least on latest nixpkgs.
    #tf = pkgs.python36Packages.tensorflowWithCuda;
    tf = pkgs.python35Packages.tensorflow;
    #tf_objdet = pkgs.python35Packages.tensorflow_object_detection.override {
    #    tensorflow = tf;
    #};
    python_with_deps = pkgs.python35.withPackages
      (ps: [ps.scipy tf ps.matplotlib ps.pandas ps.scikitlearn
            ps.Keras
            ps.easydict ps.pillow ps.pyyaml
            ps.pyqt4 # Needed only for matplotlib backend
            ps.pycallgraph ps.graphviz
            ps.jupyter
            #tf_objdet
            ]);
in stdenv.mkDerivation rec {
  name = "yazabi-tensorflow-skill-test";

  buildInputs = with pkgs; [ python_with_deps ];
}

