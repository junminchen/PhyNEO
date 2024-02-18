#!/usr/bin/env python
import sys
import jax
import jax.numpy as jnp
from jax import vmap, jit, value_and_grad
import numpy as np
from dmff.utils import jit_condition, regularize_pairs, pair_buffer_scales
from dmff.admp.pairwise import distribute_scalar, distribute_v3
from dmff.admp.spatial import pbc_shift
from dmff.utils import pair_buffer_scales, regularize_pairs
from functools import partial
import jax.nn.initializers
from jax.config import config
#config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

@jit_condition(static_argnums=())
@partial(vmap, in_axes=(0, None, None, None), out_axes=0)
def get_gto(r, rc, rs, alpha):
    gto = jnp.exp(-alpha * (r - rs)**2)
    return gto #* fc

@jit_condition(static_argnums=())
@partial(vmap, in_axes=(0, None), out_axes=0)
def cutoff_cosine(distances, cutoff):
    return jnp.square(0.5 * jnp.cos(distances * (jnp.pi / cutoff)) + 0.5)


@jit_condition(static_argnums=())
@partial(vmap, in_axes=(0, 0, None), out_axes=0)
def distribute_pair_cij(i_elem, j_elem, cij):
    return cij[i_elem, j_elem]

# calculate energy through features
@jit_condition(static_argnums=())
@partial(vmap, in_axes=(0, 0, None, None), out_axes=(0))
def get_atomic_energies(features, elem_index, params, n_layers):
    features1 = jnp.tanh(features.dot(params['w'][0][elem_index]) + params['b'][0][elem_index])
    features2 = jnp.tanh(features1.dot(params['w'][1][elem_index]) + params['b'][1][elem_index])
    features3 = features2.dot(params['w'][2][elem_index]) + params['b'][2][elem_index]
    return features3

class EANNForce:

    def __init__(self, n_elem, elem_indices, n_gto, rc, Lmax=2, beta=0.2, sizes=(20, 20), seed=12345):
        """ Constructor

        Parameters
        ----------
        n_elem: int
            Number of elements in the model.
        elem_indices: array of ints
            Element type of each atom in the system.
        n_gto: int
            Number of GTOs used in EANN.
        rc: float
            Cutoff distances, used to determine initial rs and alpha.
        Lmax: int, optional
            Largest L in angular channel. Default 2
        beta: float, optional
            beta used to determine initial \Delta rs. Default 0.2
        sizes: tupple, ints, optional
            Number of hidden neurons in the model, the length is number of layers.
            Default (20, 20)
        seed: int, optional
            Seed for random number generator, default 12345

        Examples
        ----------

        """
        self.n_elem = n_elem
        self.n_gto = n_gto
        self.rc = rc
        self.beta = beta
        self.sizes = sizes
        self.n_layers = len(sizes)
        self.Lmax = Lmax
        self.elem_indices = elem_indices
        self.n_atoms = len(elem_indices)

        # n_elements * n_features
        self.n_features = (Lmax+1) * n_gto
        cij = jnp.ones((n_elem, n_elem, Lmax+1, n_gto)) * 0.0
        rs, alpha = self.get_init_rs(n_gto, beta, rc)
        
        # initialize NN params
        key = jax.random.PRNGKey(seed)
        initializer = jax.nn.initializers.he_uniform()
        weights = []
        bias = []

        dim_in = self.n_features
        W = []
        B = []
        for i_layer in range(self.n_layers):
            dim_out = sizes[i_layer]
            key, subkey = jax.random.split(key)
            W.append(initializer(subkey, (n_elem, dim_in, dim_out)))
            B.append(jnp.zeros((n_elem, dim_out)))
            dim_in = dim_out
        key, subkey = jax.random.split(key)
        W.append(initializer(subkey, (n_elem, dim_in)))
        key, subkey = jax.random.split(key)
        B.append(jax.random.uniform(subkey, shape=(n_elem,)))

        # prepare input parameters
        # weights: weights[i_layer][n_elem, dim_in, dim_out]
        # bias: bias[i_layer][n_elem, dim_out]
        self.params = {
                'w': W,
                'b': B,
                'c': cij,
                'rs': rs,
                'alpha': alpha,
                }

        # prepare angular channels
        npara = [1]
        for i in range(1,self.Lmax+1):
            npara.append(3**i)
        self.index_para = jnp.concatenate([jnp.ones((npara[i],), dtype=int) * i for i in range(len(npara))])

        # generate get_energy
        self.get_energy = self.generate_get_energy()

        return

    def get_init_rs(self, n_gto, beta, rc):
        """
        Generate initial values for rs and alpha (exponents)

        Parameters
        ----------
        n_gto: int
            number of radial GTOs used in EANN
        beta: float
            beta used to determine initial \Delta rs. Default 0.2
        rc: float
            cutoff distance

        Returns
        ----------
        rs: 
            (n_gto, ): list of rs (for different radial channels)
        alpha:
            (n_gto, ): list of alpha
        """
        drs = rc / (n_gto - 1 + 0.3333333333)
        a = beta / drs / drs
        rs = jnp.arange(0, rc, drs)
        return rs, jnp.ones(n_gto) * a


    def get_features(self, gtos, dr, pairs, buffer_scales):
        """ Get atomic features from pairwise gto arrays
        
        Parameters
        ----------
        gtos: array, (2, n_pairs, Lmax+1, n_gtos)
            pairwise gto values, that is, 
            cij * exp(-alpha * (r-rs)**2) * 0.25*(cos(r/rc*pi) + 1)**2
        dr: array
            dr_vec for each pair, pbc shifted
        pairs: int array
            Indices of interacting pairs
        buffer_scales: float (0 or 1)
            neighbor list buffer masks

        Returns
        ----------
        features: (n_atom, n_features) array
            Atomic features

        Examples
        ----------
        """
        dr = jnp.concatenate((dr, -dr),axis=0)
        dr_norm = jnp.linalg.norm(dr, axis=1)
        pairs = jnp.concatenate((pairs,pairs[:,[1,0]]),axis=0)
        buffer_scales = jnp.concatenate((buffer_scales,buffer_scales),axis=0)
        gtos = jnp.concatenate((gtos[0, :, :, :],gtos[1, :, :, :]),axis=0)

        f_cut = cutoff_cosine(dr_norm, self.rc)
        prefacs = f_cut.reshape(1, -1)
        angular = prefacs 

        wf_terms = jnp.einsum("ijk,ji->ijk", jnp.expand_dims(gtos[:,0,:], axis=1), angular)
        totneighbour = len(pairs)
        for L in range(1,self.Lmax+1):
            prefacs = jnp.einsum("ji,ki->jki", prefacs, dr.T).reshape(-1, totneighbour)
            angular = prefacs
            wf_term = jnp.einsum("ijk,ji->ijk", jnp.expand_dims(gtos[:,L,:], axis=1), angular)
            wf_terms = jnp.concatenate((wf_terms, wf_term), axis=1)
        
        wf_terms = jnp.einsum("ijk,i->ijk", wf_terms, buffer_scales)
        wf = jnp.zeros((self.n_atoms, wf_terms.shape[1], self.n_gto))
        wf = wf.at[pairs[:,0],:,:].add(wf_terms)

        features = jnp.zeros((self.n_atoms, self.Lmax+1, self.n_gto)) 

        features = features.at[:,self.index_para,:].add(jnp.square(wf)) 
        features = features.reshape(self.n_atoms,-1)
        return features



    def generate_get_energy(self):

        @jit_condition(static_argnums=())
        def get_energy(positions, box, pairs, params):
            """ Get energy
            This function returns the EANN energy.

            Parameters
            ----------
            positions: (n_atom, 3) array
                The positions of all atoms, in cartesian
            box: (3, 3) array
                The box array, arranged in rows
            pairs: jax_md nbl index
                The neighbor list, in jax_md.partition.OrderedSparse format
            params: dict
                The parameter dictionary, including the following keys:
                c: ${c_{ij}} of all exponent prefactors, (n_elem, n_elem)
                rs: distance shifts of all radial gaussian functions, (n_gto,)
                alpha: the exponents, (n_gto,)
                w: weights of NN, list of (n_elem, dim_in, dime_out) array, with a length of n_layer
                b: bias of NN, list of (n_elem, dim_out) array, with a length of n_layer
            
            Returns:
            ----------
            energy: float or double
                EANN energy

            Examples:
            ----------
            """

            pairs = pairs[:,:2]
            pairs = regularize_pairs(pairs)
            buffer_scales = pair_buffer_scales(pairs)

            # get distances
            box_inv = jnp.linalg.inv(box)
            ri = distribute_v3(positions, pairs[:, 0])
            rj = distribute_v3(positions, pairs[:, 1])
            dr = rj - ri
            dr = pbc_shift(dr, box, box_inv)

            dr_norm = jnp.linalg.norm(dr, axis=1)

            buffer_scales2 = jnp.piecewise(buffer_scales, (dr_norm <= 4, dr_norm > 4),
                            (lambda x: jnp.array(1), lambda x: jnp.array(0)))
            buffer_scales = buffer_scales2 * buffer_scales

            rs = params['rs']
            alpha = params['alpha']

            gtos = get_gto(dr_norm, self.rc, rs, alpha)

            # element indices
            i_elem = distribute_scalar(self.elem_indices, pairs[:, 0])
            j_elem = distribute_scalar(self.elem_indices, pairs[:, 1])
            cij_per_pair = distribute_pair_cij(i_elem, j_elem, params['c'])
            cji_per_pair = distribute_pair_cij(j_elem, i_elem, params['c'])
            c_per_pair = jnp.stack((cij_per_pair, cji_per_pair), axis=0)
            gtos = c_per_pair * jnp.expand_dims(gtos, (0, 2))

            features = self.get_features(gtos, dr, pairs, buffer_scales)
        
            atomic_energies = get_atomic_energies(features, self.elem_indices, params, self.n_layers)
            return jnp.sum(atomic_energies)

        return get_energy
