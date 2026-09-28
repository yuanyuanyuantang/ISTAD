def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    print(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    print(f'  {"Seed:":<20}{args.seed:<20}{"Itr:":<20}{args.itr:<20}')
    if getattr(args, 'setting', ''):
        print(f'  {"Setting Override:":<20}{args.setting:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    print(f'  {"Features:":<20}{args.features:<20}{"Seq Len:":<20}{args.seq_len:<20}')
    print(f'  {"Enc In:":<20}{args.enc_in:<20}{"C Out:":<20}{args.c_out:<20}')
    print(f'  {"Checkpoints:":<20}{args.checkpoints}')
    print(f'  {"Eval Step:":<20}{args.istad_eval_step:<20}')
    print()

    print("\033[1m" + "Anomaly Detection Task" + "\033[0m")
    print(f'  {"Anomaly Ratio:":<20}{args.anomaly_ratio:<20}{"Threshold Source:":<20}'
          f'{args.istad_threshold_source:<20}')
    print()

    print("\033[1m" + "ISTAD Parameters" + "\033[0m")
    print(f'  {"Kernel Size:":<20}{args.istad_kernel_size:<20}{"Branch Mode:":<20}{args.istad_branch_mode:<20}')
    print(f'  {"TCN Type:":<20}{args.istad_tcn_type:<20}')
    print(f'  {"Feat GAT Dim:":<20}{args.istad_feat_gat_embed_dim:<20}{"GRU Layers:":<20}{args.istad_gru_n_layers:<20}')
    print(f'  {"GRU Hidden:":<20}{args.istad_gru_hid_dim:<20}{"Recon Layers:":<20}{args.istad_recon_n_layers:<20}')
    print(f'  {"Recon Hidden:":<20}{args.istad_recon_hid_dim:<20}{"ISTAD Dropout:":<20}{args.istad_dropout:<20}')
    print(f'  {"Recon Type:":<20}{args.istad_recon_type:<20}{"KANAD Order:":<20}{args.istad_kanad_order:<20}')
    print(f'  {"Objective:":<20}{getattr(args, "istad_objective", "reconstruct"):<20}'
          f'{"Target / Lag:":<20}{str((getattr(args, "istad_target_features", ""), getattr(args, "istad_forecast_lag", 1))):<20}')
    print(f'  {"Leaky Alpha:":<20}{args.istad_alpha:<20}{"H Init:":<20}{args.istad_h_param_init:<20}')
    print(f'  {"Hyperedges:":<20}{args.istad_n_hyperedges:<20}{"Top-K Nodes:":<20}{args.istad_k_top:<20}')
    print(f'  {"Spatial Type:":<20}{args.istad_spatial_type:<20}{"HGAT Rank:":<20}{args.istad_hgat_rank:<20}')
    print(f'  {"HGAT Projection:":<20}{getattr(args, "istad_hgat_projection", "linear"):<20}'
          f'{"Relation Mode:":<20}{getattr(args, "istad_hgat_relation_mode", "dynamic"):<20}')
    print(f'  {"KAN Grid:":<20}{args.istad_kan_grid_size:<20}{"KAN Order:":<20}{args.istad_kan_spline_order:<20}')
    print(f'  {"Arch:":<20}{getattr(args, "istad_arch", "legacy"):<20}{"Ablation:":<20}{getattr(args, "istad_ablation", "full"):<20}')
    if getattr(args, 'istad_arch', 'legacy') == 'hgst2':
        print(f'  {"Relation Dim:":<20}{args.istad_relation_dim:<20}{"HG Temp:":<20}{args.istad_hypergraph_temperature:<20}')
        print(f'  {"Temp Width:":<20}{args.istad_temporal_width:<20}{"Temp Kernel:":<20}{args.istad_temporal_kernel_size:<20}')
        print(f'  {"T Grid:":<20}{args.istad_temporal_grid_size:<20}{"T Order:":<20}{args.istad_temporal_spline_order:<20}')
        print(f'  {"Mod Rank:":<20}{args.istad_modulation_rank:<20}{"Mod Scale:":<20}{args.istad_modulation_scale:<20}')
        print(f'  {"S Grid:":<20}{args.istad_spatial_grid_size:<20}{"S Order:":<20}{args.istad_spatial_spline_order:<20}')
        print(f'  {"Decoder Hidden:":<20}{args.istad_decoder_hidden:<20}')
    if getattr(args, 'istad_arch', 'legacy') in {'v7', 'v71'}:
        print(f'  {"V7 relation dim:":<20}{args.istad_v7_relation_dim:<20}'
              f'{"Use hypergraph:":<20}{args.istad_v7_use_hypergraph:<20}')
        print(f'  {"Prior ridge/topk:":<20}{str((args.istad_v7_prior_ridge, args.istad_v7_prior_topk)):<20}')
        print(f'  {"Prior strength:":<20}{args.istad_v7_prior_strength:<20}'
              f'{"Prior floor/lambda:":<20}{str((args.istad_v7_prior_floor, args.istad_v7_prior_lambda)):<20}')
        print(f'  {"Relation score wt:":<20}{args.istad_v7_relation_score_weight:<20}'
              f'{"Entity calibration:":<20}{args.istad_v7_entity_calibration:<20}')
        print(f'  {"Entity context:":<20}{getattr(args, "istad_entity_context", 0):<20}'
              f'{"Score mode:":<20}{args.istad_score_mode:<20}')
        if getattr(args, 'istad_arch', 'legacy') == 'v71':
            print(f'  {"Graph gate init:":<20}{args.istad_v71_graph_gate_init:<20}'
                  f'{"Prior gate init:":<20}{args.istad_v71_prior_gate_init:<20}')
    print(f'  {"Explainability:":<20}{args.istad_enable_explain:<20}{"Explain Batches:":<20}{args.istad_explain_max_batches:<20}')
    print(f'  {"Explain Points:":<20}{args.istad_explain_max_points:<20}{"Explain TopK:":<20}{args.istad_explain_topk:<20}')
    print(f'  {"SmoothGrad:":<20}{args.istad_use_smoothgrad:<20}{"SG Samples:":<20}{args.istad_smoothgrad_samples:<20}')
    print(f'  {"SG Noise:":<20}{args.istad_smoothgrad_noise:<20}{"Quad Mode:":<20}{args.istad_quadrant_threshold_mode:<20}')
    print(f'  {"Quad Value:":<20}{args.istad_quadrant_threshold_value:<20}{"Err Agg:":<20}{args.istad_quadrant_agg_error:<20}')
    print(f'  {"Sal Agg:":<20}{args.istad_quadrant_agg_saliency:<20}')
    if bool(int(getattr(args, 'istad_denoise', 0) or 0)):
        print(f'  {"V3 Denoise:":<20}{args.istad_denoise:<20}{"Evidence Head:":<20}{args.istad_evidence_head:<20}')
        print(f'  {"Clean / Denoise:":<20}{str((args.istad_clean_lambda, args.istad_denoise_lambda)):<20}'
              f'{"Evidence Weight:":<20}{args.istad_evidence_lambda:<20}')
        print(f'  {"Frequency Weight:":<20}{args.istad_frequency_lambda:<20}')
        print(f'  {"Point Evidence:":<20}{args.istad_point_evidence_lambda:<20}')
        print(f'  {"Corrupt P/T/C:":<20}{str((args.istad_corrupt_prob, args.istad_corrupt_time_ratio, args.istad_corrupt_channel_ratio)):<20}'
              f'{"Corrupt Scale:":<20}{args.istad_corrupt_scale:<20}')
        print(f'  {"Corrupt Modes:":<20}{args.istad_corrupt_modes:<20}'
              f'{"Target-only corrupt:":<20}{getattr(args, "istad_corrupt_target_only", 0):<20}')
        print(f'  {"Score Mode:":<20}{args.istad_score_mode:<20}{"Score Agg / K:":<20}'
              f'{str((args.istad_score_aggregate, args.istad_score_topk)):<20}')
        print(f'  {"Evidence Map:":<20}{args.istad_evidence_transform:<20}{"Dump Scores:":<20}'
              f'{args.istad_dump_scores:<20}')
        print(f'  {"Strict Val:":<20}{args.istad_holdout_val:<20}{"Hide Test Loss:":<20}'
              f'{args.istad_no_test_during_train:<20}')
        print(f'  {"Entity Aware:":<20}{args.istad_entity_aware:<20}')
    if str(getattr(args, 'istad_score_mode', '')).lower() in {
        'innovation', 'innovation_fused', 'innovation_hgat'
    }:
        print(f'  {"Innovation lag/ridge:":<20}'
              f'{str((args.istad_innovation_lag, args.istad_innovation_ridge)):<20}'
              f'{"Innovation pool:":<20}{args.istad_innovation_pool:<20}')
        print(f'  {"Degenerate cutoff:":<20}{args.istad_innovation_degenerate_cutoff:<20}'
              f'{"Scale floor:":<20}{args.istad_innovation_scale_floor:<20}')
        print(f'  {"Evidence tie-break:":<20}{args.istad_innovation_fusion_weight:<20}'
              f'{"Fusion train ECDF:":<20}{getattr(args, "istad_fusion_recalibrate", 0):<20}')
        if str(getattr(args, 'istad_score_mode', '')).lower() == 'innovation_hgat':
            print(f'  {"HGAT input:":<20}'
                  f'{getattr(args, "istad_hgat_input", "raw"):<20}')
            print(f'  {"HGAT fusion:":<20}'
                  f'{getattr(args, "istad_hgat_fusion_strategy", "rank_tiebreak"):<20}'
                  f'{"HGAT max weight:":<20}'
                  f'{getattr(args, "istad_hgat_fusion_max_weight", 0.2):<20}')
            print(f'  {"HGAT ref/tail:":<20}'
                  f'{str((getattr(args, "istad_hgat_reliability_reference_fraction", 0.8), getattr(args, "istad_hgat_reliability_tail_probability", 0.01))):<20}'
                  f'{"HGAT inflation:":<20}'
                  f'{getattr(args, "istad_hgat_reliability_inflation_limit", 1.25):<20}')
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Patience:":<20}{args.patience:<20}')
    print(f'  {"Learning Rate:":<20}{args.learning_rate:<20}{"Lradj:":<20}{args.lradj:<20}')
    print(f'  {"Des:":<20}{args.des:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    print()
