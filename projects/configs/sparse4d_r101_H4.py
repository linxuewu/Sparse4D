_base_ = [
    './sparse4d_r101_H1.py'
]

H = 4
max_queue_length = H - 1

model = dict(
    head=dict(
        deformable_model=dict(
            temporal_fusion_module=dict(
                type="LinearFusionModule",
            )
        ),
        instance_bank=dict(max_queue_length=max_queue_length),
    )
)

data = dict(
    train=dict(
        max_interval=2,
        fix_interval=True,
        max_time_interval=5,
        seq_frame=max_queue_length,
    )
)

'''
mAP: 0.4409
mATE: 0.6282
mASE: 0.2721
mAOE: 0.3853
mAVE: 0.2922
mAAE: 0.1888
NDS: 0.5438
Eval time: 235.2s

Per-class results:
Object Class	AP	ATE	ASE	AOE	AVE	AAE
car	0.633	0.432	0.146	0.064	0.225	0.183
truck	0.364	0.685	0.201	0.087	0.262	0.207
bus	0.432	0.770	0.215	0.096	0.589	0.238
trailer	0.198	1.035	0.281	0.516	0.298	0.139
construction_vehicle	0.120	0.956	0.471	1.059	0.116	0.345
pedestrian	0.530	0.588	0.289	0.398	0.308	0.150
motorcycle	0.458	0.600	0.254	0.439	0.363	0.222
bicycle	0.403	0.491	0.267	0.671	0.176	0.026
traffic_cone	0.674	0.324	0.311	nan	nan	nan
barrier	0.597	0.400	0.286	0.139	nan	nan
'''
