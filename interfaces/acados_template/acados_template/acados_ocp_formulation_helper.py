from typing import Union

import numpy as np
import casadi

from .acados_ocp import AcadosOcp, AcadosModel


def g_lin_constraint(ocp: AcadosOcp, c_row: np.ndarray, d_row: np.ndarray, lower_upper: Union[float, tuple],
                     sq_penalty: Union[float, tuple] = 0, lin_penalty: Union[float, tuple] = 0):
    """
    Add a path linear constraints with lower and upper bound.
    Generates the lower, upper and slack variables, and optionally the idxXXX selection vector for soft constraints.
    :param ocp: the ocp to add the constraint to
    :param c_row: linear coefficients for states
    :param d_row: linear coefficients for controls
    :param lower_upper: either a tuple to specify (lower, upper) or a single value to set bounds: (-value, value)
    :param sq_penalty: quadratic penalty on the slack variables, either tuple (s_l, s_u) or single value for both
    :param lin_penalty: linear penalty on the slack variables, either tuple (s_l, s_u) or single value for both
    """
    if c_row.ndim == 1:
        c_row = np.expand_dims(c_row, 0)
    elif d_row.ndim == 1:
        d_row = np.expand_dims(d_row, 0)
    assert c_row.shape[1] == get_nx(ocp) and d_row.shape[1] == get_nu(ocp)
    if ocp.constraints.C.shape[1] != get_nx(ocp):
        ocp.constraints.C = c_row
        ocp.constraints.D = d_row
    else:
        ocp.constraints.C = np.concatenate([ocp.constraints.C, c_row], axis=0)
        ocp.constraints.D = np.concatenate([ocp.constraints.D, d_row], axis=0)
    s_idx = get_nsbu(ocp) + get_nsbx(ocp) + get_nsg(ocp)
    _add_constraint_bounds(ocp, "g", s_idx, lower_upper, sq_penalty=sq_penalty, lin_penalty=lin_penalty)


def end_g_lin_constraint(ocp: AcadosOcp, c_row: np.ndarray, d_row: np.ndarray, lower_upper: Union[float, tuple],
                         sq_penalty: Union[float, tuple] = 0, lin_penalty: Union[float, tuple] = 0):
    """
    Add a terminal linear constraints with lower and upper bound.
    Generates the lower, upper and slack variables, and optionally the idxXXX selection vector for soft constraints.
    :param ocp: the ocp to add the constraint to
    :param c_row: linear coefficients for states
    :param d_row: linear coefficients for controls
    :param lower_upper: either a tuple to specify (lower, upper) or a single value to set bounds: (-value, value)
    :param sq_penalty: quadratic penalty on the slack variables, either tuple (s_l, s_u) or single value for both
    :param lin_penalty: linear penalty on the slack variables, either tuple (s_l, s_u) or single value for both
    """
    if c_row.ndim == 1:
        c_row = np.expand_dims(c_row, 0)
    elif d_row.ndim == 1:
        d_row = np.expand_dims(d_row, 0)
    assert c_row.shape[1] == get_nx(ocp) and d_row.shape[1] == get_nu(ocp)
    if ocp.constraints.C.shape[1] != get_nx(ocp):
        ocp.constraints.C_e = c_row
        ocp.constraints.D_e = d_row
    else:
        ocp.constraints.C_e = np.concatenate([ocp.constraints.C_e, c_row], axis=0)
        ocp.constraints.D_e = np.concatenate([ocp.constraints.D_e, d_row], axis=0)
    s_idx = get_nsbx_e(ocp) + get_nsg_e(ocp)
    _add_constraint_bounds(ocp, "g_e", s_idx, lower_upper, sq_penalty=sq_penalty, lin_penalty=lin_penalty)


def h_nl_constraint(ocp, nl_expr, lower_upper: Union[float, tuple],
                    sq_penalty: Union[float, tuple] = 0, lin_penalty: Union[float, tuple] = 0):
    """
    Add a path non-linear constraints with lower and upper bound.
    Generates the lower, upper and slack variables, and optionally the idxXXX selection vector for soft constraints.
    :param ocp: the ocp to add the constraint to
    :param nl_expr: the casadi non-linear expression for the non-linear constraint
    :param lower_upper: either a tuple to specify (lower, upper) or a single value to set bounds: (-value, value)
    :param sq_penalty: quadratic penalty on the slack variables, either tuple (s_l, s_u) or single value for both
    :param lin_penalty: linear penalty on the slack variables, either tuple (s_l, s_u) or single value for both
    """
    if ocp.model.con_h_expr is None:
        ocp.model.con_h_expr = casadi.vertcat(nl_expr)
    else:
        ocp.model.con_h_expr = casadi.vertcat(ocp.model.con_h_expr, nl_expr)
    s_idx = get_nsbu(ocp) + get_nsbx(ocp) + get_nsg(ocp) + get_nsh(ocp)
    _add_constraint_bounds(ocp, "h", s_idx, lower_upper, sq_penalty=sq_penalty, lin_penalty=lin_penalty)


def end_h_nl_constraint(ocp: AcadosOcp, nl_expr, lower_upper: Union[float, tuple],
                        sq_penalty: Union[float, tuple] = 0, lin_penalty: Union[float, tuple] = 0):
    """
    Add a terminal non-linear constraints with lower and upper bound.
    Generates the lower, upper and slack variables, and optionally the idxXXX selection vector for soft constraints.
    :param ocp: the ocp to add the constraint to
    :param nl_expr: the casadi non-linear expression for the non-linear constraint
    :param lower_upper: either a tuple to specify (lower, upper) or a single value to set bounds: (-value, value)
    :param sq_penalty: quadratic penalty on the slack variables, either tuple (s_l, s_u) or single value for both
    :param lin_penalty: linear penalty on the slack variables, either tuple (s_l, s_u) or single value for both
    """
    if ocp.model.con_h_expr_e is None:
        ocp.model.con_h_expr_e = casadi.vertcat(nl_expr)
    else:
        ocp.model.con_h_expr_e = casadi.vertcat(ocp.model.con_h_expr_e, nl_expr)
    s_idx = get_nsbx_e(ocp) + get_nsg_e(ocp) + get_nsh_e(ocp)
    _add_constraint_bounds(ocp, "h_e", s_idx, lower_upper, sq_penalty=sq_penalty, lin_penalty=lin_penalty)


def x_bound(ocp: AcadosOcp, name: str, lower_upper: Union[float, tuple],
            sq_penalty: Union[float, tuple] = 0, lin_penalty: Union[float, tuple] = 0):
    """
    Add path bounds on a state variable.
    Generates the lower, upper and slack variables, and optionally the idxXXX selection vector for soft constraints.
    :param ocp: the ocp to add the constraint to
    :param name: the casadi name of the variable to constrain
    :param lower_upper: either a tuple to specify (lower, upper) or a single value to set bounds: (-value, value)
    :param sq_penalty: quadratic penalty on the slack variables, either tuple (s_l, s_u) or single value for both
    :param lin_penalty: linear penalty on the slack variables, either tuple (s_l, s_u) or single value for both
    """
    x_idx = get_symbol_idx(ocp.model.x, name)
    ocp.constraints.idxbx = np.concatenate([ocp.constraints.idxbx, [x_idx]])
    s_idx = get_nsbu(ocp) + get_nsbx(ocp)
    _add_constraint_bounds(ocp, "bx", s_idx, lower_upper, sq_penalty=sq_penalty, lin_penalty=lin_penalty)


def end_x_bound(ocp: AcadosOcp, name: str, lower_upper: Union[float, tuple],
                sq_penalty: Union[float, tuple] = 0, lin_penalty: Union[float, tuple] = 0):
    """
    Add terminal bounds on a state variable.
    Generates the lower, upper and slack variables, and optionally the idxXXX selection vector for soft constraints.
    :param ocp: the ocp to add the constraint to
    :param name: the casadi name of the variable to constrain
    :param lower_upper: either a tuple to specify (lower, upper) or a single value to set bounds: (-value, value)
    :param sq_penalty: quadratic penalty on the slack variables, either tuple (s_l, s_u) or single value for both
    :param lin_penalty: linear penalty on the slack variables, either tuple (s_l, s_u) or single value for both
    """
    x_idx = get_symbol_idx(ocp.model.x, name)
    ocp.constraints.idxbx_e = np.concatenate([ocp.constraints.idxbx_e, [x_idx]])
    s_idx = get_nsbx_e(ocp)
    _add_constraint_bounds(ocp, "bx_e", s_idx, lower_upper, sq_penalty=sq_penalty, lin_penalty=lin_penalty)


def u_bound(ocp: AcadosOcp, name: str, lower_upper: Union[float, tuple],
            sq_penalty: Union[float, tuple] = 0, lin_penalty: Union[float, tuple] = 0):
    """
    Add bounds on a control variable.
    Generates the lower, upper and slack variables, and optionally the idxXXX selection vector for soft constraints.
    :param ocp: the ocp to add the constraint to
    :param name: the casadi name of the variable to constrain
    :param lower_upper: either a tuple to specify (lower, upper) or a single value to set bounds: (-value, value)
    :param sq_penalty: quadratic penalty on the slack variables, either tuple (s_l, s_u) or single value for both
    :param lin_penalty: linear penalty on the slack variables, either tuple (s_l, s_u) or single value for both
    """
    u_idx = get_symbol_idx(ocp.model.u, name)
    ocp.constraints.idxbu = np.concatenate([ocp.constraints.idxbu, [u_idx]])
    s_idx = get_nsbu(ocp)
    _add_constraint_bounds(ocp, "bu", s_idx, lower_upper, sq_penalty=sq_penalty, lin_penalty=lin_penalty)


def _add_constraint_bounds(ocp: AcadosOcp, id_: str, slack_offset: int, lower_upper: Union[float, tuple],
                           sq_penalty: Union[float, tuple] = 0, lin_penalty: Union[float, tuple] = 0):
    """
    Utility function to manage the common elements of (soft) constraint creation.
    Generates the lower, upper and slack variables, and optionally the idxXXX selection vector for soft constraints.
    :param ocp: the ocp to add the constraint to
    :param id_: the name of the constraint function, i.e. bx, bu, h, g, and X_e
    :param slack_offset: optionally, the index of the soft constraint in the slack cost vectors
    :param lower_upper: either a tuple to specify (lower, upper) or a single value to set bounds: (-value, value)
    :param sq_penalty: quadratic penalty on the slack variables, either tuple (s_l, s_u) or single value for both
    :param lin_penalty: linear penalty on the slack variables, either tuple (s_l, s_u) or single value for both
    """
    is_terminal = "_e" in id_
    lower, upper = lower_upper if isinstance(lower_upper, tuple) else (-lower_upper, lower_upper)
    sl_quad, su_quad = sq_penalty if isinstance(sq_penalty, tuple) else (sq_penalty, sq_penalty)
    sl_lin, su_lin = lin_penalty if isinstance(lin_penalty, tuple) else (lin_penalty, lin_penalty)
    setattr(ocp.constraints, "l" + id_, np.concatenate([getattr(ocp.constraints, "l" + id_), [lower]]))
    setattr(ocp.constraints, "u" + id_, np.concatenate([getattr(ocp.constraints, "u" + id_), [upper]]))
    if sl_quad > 0 or su_quad > 0 or sl_lin > 0 or su_lin > 0:
        check_slack_dimensions(ocp, path=not is_terminal, terminal=is_terminal)
        constraint_idx = getattr(ocp.constraints, "u" + id_).shape[0] - 1
        setattr(ocp.constraints, "idxs" + id_, np.concatenate([getattr(ocp.constraints, "idxs" + id_), [constraint_idx]]))
        suffix = "_e" if is_terminal else ""
        iZl, iZu, izl, izu = "Zl" + suffix, "Zu" + suffix, "zl" + suffix, "zu" + suffix
        Zl, Zu, zl, zu = getattr(ocp.cost, iZl), getattr(ocp.cost, iZu), getattr(ocp.cost, izl), getattr(ocp.cost, izu)
        setattr(ocp.cost, iZl, np.concatenate([Zl[:slack_offset], [sl_quad], Zl[slack_offset:]]))
        setattr(ocp.cost, iZu, np.concatenate([Zu[:slack_offset], [su_quad], Zu[slack_offset:]]))
        setattr(ocp.cost, izl, np.concatenate([zl[:slack_offset], [sl_lin], zl[slack_offset:]]))
        setattr(ocp.cost, izu, np.concatenate([zu[:slack_offset], [su_lin], zu[slack_offset:]]))
        check_slack_dimensions(ocp, path=not is_terminal, terminal=is_terminal)


def get_nx(ocp: AcadosOcp):
    """ number of states """
    return int(ocp.model.x.size()[0])


def get_nu(ocp: AcadosOcp):
    """ number of controls """
    return int(ocp.model.u.size()[0])


def get_nsbu(ocp: AcadosOcp):
    """ number of slack variables for bounds on controls """
    return int(ocp.constraints.idxsbu.shape[0])


def get_nsbx(ocp: AcadosOcp):
    """ number of slack variables for bounds on states """
    return int(ocp.constraints.idxsbx.shape[0])


def get_nsbx_e(ocp: AcadosOcp):
    """ number of slack variables for bounds on terminal states """
    return int(ocp.constraints.idxsbx_e.shape[0])


def get_nsg(ocp: AcadosOcp):
    """ number of slack variables for linear constraints """
    return int(ocp.constraints.idxsg.shape[0])


def get_nsg_e(ocp: AcadosOcp):
    """ number of slack variables for linear constraints on terminal state and controls """
    return int(ocp.constraints.idxsg_e.shape[0])


def get_nsh(ocp: AcadosOcp):
    """ number of slack variables for non-linear constraints """
    return int(ocp.constraints.idxsh.shape[0])


def get_nsh_e(ocp: AcadosOcp):
    """ number of slack variables for non-linear constraints of the terminal state """
    return int(ocp.constraints.idxsh_e.shape[0])


def auto_xdot(model_x):
    """
        For a vector of scalar casadi state variables, generate a corresponding vector of {name}_dot casadi symbols.
    """
    xdot = casadi.vertcat([])
    for i in range(model_x.size()[0]):
        xdot = casadi.vertcat(xdot, casadi.MX.sym(model_x[i].name() + "_dot"))
    return xdot


def get_state_var(ocp_model: Union[AcadosOcp, AcadosModel], name):
    if isinstance(ocp_model, AcadosOcp):
        ocp_model = ocp_model.model
    return get_symbol(ocp_model.x, name)


def get_control_var(ocp_model: Union[AcadosOcp, AcadosModel], name):
    if isinstance(ocp_model, AcadosOcp):
        ocp_model = ocp_model.model
    return get_symbol(ocp_model.u, name)


def get_symbol(symbol_vector, name):
    """
        Find a casadi symbol by its name in a vertcat (vertical concatenation) vector of symbols.
        This method can also find vector-shaped symbols that span across a part of the full symbol vector.
    """
    idx = get_symbol_idx(symbol_vector, name)
    if idx is None:
        return None
    elif isinstance(idx, tuple):
        return symbol_vector[idx[0]:idx[1]]
    else:
        return symbol_vector[idx]


def get_symbol_idx(symbol_vector, name):
    """
        Find the index of a casadi symbol by its name in a vertcat (vertical concatenation) of symbols.
        If the symbol is a vector instead of a scalar, this method returns the index range of the symbol.
    """
    v_len = symbol_vector.size()[0]
    for i in range(v_len):
        try:
            if symbol_vector[i].name() == name:
                return i
        except RuntimeError:
            for j in range(i, v_len + 1):
                try:
                    if symbol_vector[i:j].name() == name:
                        return i, j
                except RuntimeError:
                    pass
    return None


def get_symbols_with_positions(symbol_vector):
    """
        Find all symbols in a casadi symbol vector and return a list with tuples (name, i, j) where the symbol is located at symbol_vector[i:j].
    """
    symbols = []
    for i in range(symbol_vector.size()[0]):
        for j in range(i, symbol_vector.size()[0] + 1):
            try:
                name = symbol_vector[i:j].name()
                symbols.append((name, i, j))
            except RuntimeError:
                pass
    return symbols


def check_slack_dimensions(ocp: AcadosOcp, path=False, terminal=False):
    """
    For soft path or terminal constraints and slack cost vectors, check if their dimensions match.
    """
    error = get_slack_dim_mismatch_error(ocp, path=path, terminal=terminal)
    if error is not None:
        raise RuntimeError(error)


def get_slack_dim_mismatch_error(ocp, path=False, terminal=False):
    """
    Check if any of the slack cost vectors does not have the same dimension as the number of soft constraints.
    If there is a mismatch, return an error string describing the probem
    """
    zs = ["Zl", "Zu", "zl", "zu"]
    mismatch = []
    if path:
        num_slack = get_nsg(ocp) + get_nsh(ocp) + get_nsbx(ocp) + get_nsbu(ocp)
        mismatch = [z_name for z_name in zs if getattr(ocp.cost, z_name).shape[0] != num_slack]
    if terminal and len(mismatch) == 0:
        num_slack = get_nsg_e(ocp) + get_nsh_e(ocp) + get_nsbx_e(ocp)
        mismatch = [z_name + "_e" for z_name in zs if getattr(ocp.cost, z_name + "_e").shape[0] != num_slack]
    if len(mismatch) == 0:
        return None
    else:
        return "Mismatch: slack cost vector and number of soft constraints: |{z_name}|={nz} vs. |soft constr|={ns}" \
               "".format(z_name=mismatch[0], nz=getattr(ocp.cost, mismatch[0]).shape[0], ns=num_slack)
