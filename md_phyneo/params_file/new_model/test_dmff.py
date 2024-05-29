#!/usr/bin/env python3
import os
import sys
import driver
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
import openmm
from openmm import *
from openmm.app import *
from openmm.unit import *
import dmff
from dmff.api import Hamiltonian
from dmff.common import nblist
from dmff.utils import jit_condition
from graph import TopGraph, from_pdb
from gnn import MolGNNForce
from eann import EANNForce
import pickle
from jax.config import config


config.update("jax_enable_x64", True)
