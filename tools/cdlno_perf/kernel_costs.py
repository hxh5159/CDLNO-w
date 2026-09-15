"""Independent closed-form cross-check of the live operation audit."""

def formulas(b,n,d,m,l,r,conv=False):
    matched=b*l*(8*n*d*d+15*m*d*d+4*n*m*d+2*m*m*d)
    if conv:matched+=9*b*l*n*d*d
    removed=b*l*(4*m*d*d+2*m*m*d)
    history=b*m*d*r*(l-1)*(l+6)//2
    denominator=b*m*r*l*(l-1)//2
    return dict(matched_core_main_mac=matched,removed_sa_mac=removed,
                kernel_history_main_mac=history,kernel_denominator_mac=denominator,
                kernel_core_main_mac=matched-removed+history,
                principal_parameter_reduction=4*l*d*d-2*(l-1)*d*r)


def check_cost(model,case,live):
    c=model.config
    f=formulas(case.B,case.N,c.d,c.M,c.L,getattr(c,'kernel_rank',16),c.point_module=='conv_ffn')
    name=c.family
    expected=f['matched_core_main_mac']
    if name=='kcdno':
        expected-=f['removed_sa_mac']
        if c.history_mode=='all':expected+=f['kernel_history_main_mac']+f['kernel_denominator_mac']
    # Lift/time and task output projections are outside the core formula.
    outer=sum(v for k,v in live['matrix_macs_by_component'].items() if k in ('stem_and_time','output_head'))
    assert live['matrix_macs']-outer==expected,(live['matrix_macs'],outer,expected)
    reads=c.L*(c.L-1)//2 if name=='kcdno' and c.history_mode=='all' else 0
    assert live['counts']['kernel_reads']==reads
    assert live['counts']['latent_sa']==(c.L if name=='lrsa_matched' else 0)
    assert live['counts']['down_bridge']==c.L and live['counts']['up_readout']==c.L
    assert live['counts']['front_latent_ffn']==2*c.L and live['counts']['point_modules']==c.L
    expected_cache=case.B*(c.L-1)*(c.kernel_rank*c.d+c.kernel_rank)*4 if reads else 0
    assert live['storage']['kernel_inference_summaries_bytes']==expected_cache
    return dict(**f,measured_core_matrix_mac=expected,outer_matrix_mac=outer,
        verified=True,convention='Main formulas omit denominator contraction and pointwise/norm/phi/depth work; denominator included in live MAC. Depth dot/fusion/softmax, mass sum, gates and copies separately inventoried.')
