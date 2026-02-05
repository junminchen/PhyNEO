from sys import *
#******** import parent classes
from shared.MM_class_base import *
from MM_classes_FV import *
#******** must have QM/MM module
try:
    from MM_classes import *
except ImportError:
    print( 'Error in importing QM/MM MM class---cannot use MM_classes_FV_QMMM without this!')
    sys.exit()
#*****************


#************************************************************************
# This child class combines child MM classes for Fixed-Voltage and QM/MM 
# inheritance diagram is:
#                     MM_base
#                    -      -
#                  -          -
#               MM_QMMM     MM_FixedVoltage
#                   _        _
#                     _     _
#               MM_FixedVoltage_QMMM
#
# note that using super().__init__() in all child classes ensures that
# MM_base.__init__() isn't doubly called.
#**************************************************************************
class MM_FixedVoltage_QMMM(MM_QMMM, MM_FixedVoltage):
    # required input: 1) list of pdb files, 2) list of residue xml files, 3) list of force field xml files.
    def __init__( self , pdb_list , residue_xml_list , ff_xml_list , **kwargs  ):

        # constructor runs through MM_QMMM and MM_FixedVoltage ...
        super().__init__( pdb_list , residue_xml_list , ff_xml_list , **kwargs )

