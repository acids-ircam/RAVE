from pathlib import Path
import shutil
import os, re
import sys

import gin
import torch
from absl import flags, app, logging

import acids_dataset as ad
from acids_dataset import transforms as adt, features as adf
from acids_dataset.features import ModuleEmbedding

try:
    import rave
except:
    import sys, os 
    sys.path.append(os.path.abspath('.'))
    import rave

try:
    from after.diffusion import RectifiedFlow, EDM
    from after.diffusion.utils import collate_fn
except ImportError as e:
    logging.error(f"Could not import AFTER ({e}). Install it with: pip install \"acids-rave[after]\"")
    sys.exit(1)

import rave.dataset
import rave.scripting


def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name


after_regexp = r'checkpoint(\d+)\D*\.pt'
def find_after_model(ckpt_path):
    if not os.path.exists(ckpt_path): raise FileNotFoundError(ckpt_path)
    if os.path.isdir(ckpt_path):
        valid_checkpoints = filter(lambda x: os.path.splitext(x)[1] == ".pt", os.listdir(ckpt_path))
        valid_checkpoints = filter(lambda x: re.search(after_regexp, x), valid_checkpoints)
        valid_checkpoints = sorted(valid_checkpoints, key=lambda x: int(re.match(after_regexp, x).groups()[0]))
        if len(valid_checkpoints) == 0: raise FileNotFoundError('No valid checkpoint found in %s'%ckpt_path)
        ckpt_dir = ckpt_path
        ckpt_name = valid_checkpoints[-1]
    else:
        assert re.search(after_regexp, ckpt_path), "%s is not a valid checkpoint (cannot take restart step)"
        ckpt_name = os.path.basename(ckpt_path)
        ckpt_dir = os.path.abspath(os.path.join(ckpt_path, ".."))
    restart_step = int(re.match(after_regexp, ckpt_name).groups()[0])
    return ckpt_name, ckpt_dir, restart_step



def check_rave_model(model_path, ckpt_name="last.ckpt"):
    """raises if model_path is not a RAVE model: a RAVE run (run / version folder, .ckpt, with
    its config.gin) or a model exported by `rave export` (.ts)."""
    from rave.uncompiling import is_scripted_rave
    hint = ("rave train_after only works on RAVE models (run folder, .ckpt or .ts exported by rave export). "
            "To train AFTER on other codecs, use AFTER directly: https://github.com/acids-ircam/AFTER")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    if os.path.splitext(model_path)[1] == ".ts":
        try:
            ts = torch.jit.load(model_path, map_location="cpu")
        except Exception as e:
            raise ValueError(f"could not load {model_path} ({e}). {hint}") from e
        if not is_scripted_rave(ts):
            raise ValueError(f"{model_path} is not a RAVE export ({getattr(ts, 'original_name', type(ts).__name__)}). {hint}")
        return
    if os.path.isfile(model_path) and os.path.splitext(model_path)[1] != ".ckpt":
        raise ValueError(f"{model_path} is neither a RAVE run, a .ckpt nor a .ts. {hint}")
    ckpt = rave.core.search_for_run(model_path, name=ckpt_name) or rave.core.search_for_run(model_path, name=None)
    if ckpt is None:
        raise FileNotFoundError(f"no RAVE checkpoint found in {model_path}. {hint}")
    config = rave.core.search_for_config(str(model_path))
    if config is None or "RAVE" not in Path(config).read_text():
        raise ValueError(f"no RAVE configuration (config.gin) found for {model_path}. {hint}")


def parse_flags():

    flags.DEFINE_string('name', None, help='Name of the run')
    flags.DEFINE_string('model', default=None, required=True, help="pretrained RAVE model: exported .ts, run folder, version folder or .ckpt")
    flags.DEFINE_string('ckpt_name', default="last.ckpt", help="RAVE checkpoint name when --model is a run folder (falls back to the most recent one)")
    flags.DEFINE_bool('ema_weights', default=False, help="use the EMA weights of the RAVE run, if available")
    flags.DEFINE_string('config', default="base", help="AFTER model config (rectified or edm)")
    flags.DEFINE_string('arch', default="rectified", help="AFTER model config (rectified or edm)")
    flags.DEFINE_string('out_path', default="runs/", help="out directory path")
    flags.DEFINE_string('embedding_name', default=None, help="sets embedding name (default : RAVE checkpoint name)")
    flags.DEFINE_bool('force', default=False, help="Forces re-processing of augmentations & features")

    # Augmentation-related keywords 
    flags.DEFINE_bool('force_augmentation', default=False, help="Forces re-processing of augmentations")
    flags.DEFINE_integer('n_augmentations', default=4, help="number of data augmentations for AFTER training (default: 4)")
    flags.DEFINE_bool('stretch', default=True, help="augments data with time stretching (default : True)")
    flags.DEFINE_list('stretch_range', default='0.7,1.6', help="time-stretch range (default: 0.7,1.6)")
    flags.DEFINE_bool('shift', default=True, help="augments data with pitch-shifting (default : True)")
    flags.DEFINE_list('shift_range', default='-6,6', help="pitch shift range (default: -6,6)")

    # Feature-related keywords
    flags.DEFINE_string('structure', default=None, help="Information from strcture (midi or beat).")
    flags.DEFINE_bool('force_features', default=False, help="Forces re-processing of features")
    flags.DEFINE_bool('midi', default=False, help="enables MIDI extraction from MIDI files (default: False)")
    flags.DEFINE_bool('beat', default=False, help="enables beat extraction from audio files (default: False)")

    # Data-related key words
    flags.DEFINE_multi_string('db_path', default=None, required=True, help="Preprocessed dataset paths")
    flags.DEFINE_multi_float('db_weights', default=None, required=False, help="Weights for dataset balancing")
    flags.DEFINE_string('db_partition', default=None, help="Fetches given partition in dataset(s)")
    flags.DEFINE_integer('db_max_size', default=None, help="Fetches given partition in dataset(s)")
    flags.DEFINE_boolean('use_cache_for_multi', default=True, required=False, help="Use cache for multi-dataset random sampling (only for multi dataset training)")

    # Training related keywords
    flags.DEFINE_string('device', default="cpu", help='device to use (default : cpu, can be set to cuda:0, cuda:1, mps...)')
    flags.DEFINE_integer('batch', 32, help="batch size")
    flags.DEFINE_integer('n_signal', None, help="chunk size (default: chunk_size // 2)")
    flags.DEFINE_string('resume', default=None, help="AFTER run folder or checkpoint to resume")
    flags.DEFINE_string('ckpt', default=None, help="(deprecated, use --resume) AFTER checkpoint to resume")
    flags.DEFINE_float('fidelity', default=0.99, help="fidelity defining the latent size when --model is a RAVE run (ignored for .ts)")
    flags.DEFINE_integer('latent_size', default=None, help="explicit latent size when --model is a RAVE run (overrides --fidelity)")
    flags.DEFINE_float('adv_weight', default=0.05, help="adversarial weight for disentanglement (default=0.05)")
    flags.DEFINE_integer('workers',
                        default=8,
                        help='Number of workers to spawn for dataset loading')
    flags.DEFINE_integer('val_every', None, help='validation (and audio logging) every n steps (default: AFTER config, 10k)')
    flags.DEFINE_integer('save_every',
                        None,
                        help='save a checkpoint every n steps (default: AFTER config, 100k)')
    flags.DEFINE_multi_string('override', default=[], help='Override gin binding')
    flags.DEFINE_bool('derivative',
                    default=False,
                    help='Train RAVE on the derivative of the signal')
    flags.DEFINE_bool('normalize',
                    default=False,
                    help='Train RAVE on normalized signals')
    flags.DEFINE_bool('smoke_test', 
                    default=False, 
                    help="runs in smoke test (no training)")



def main(argv):

    FLAGS = flags.FLAGS
    device = torch.device(FLAGS.device)
    if FLAGS.structure == "midi":
        logging.info('structure with MIDI only works with midi architecture ; changing')
        FLAGS.config = "midi"      
    if FLAGS.config == "midi" and FLAGS.structure != "midi": 
        logging.info('structure with MIDI only works with midi architecture ; changing')
        FLAGS.structure = "midi"


    # AFTER run folder (resolved first: RAVE checkpoints are exported into it)
    resume = FLAGS.resume or FLAGS.ckpt
    if FLAGS.ckpt is not None and FLAGS.resume is None:
        logging.warning("--ckpt is deprecated, use --resume")
    if resume is not None:
        after_ckpt_name, model_dir, restart_step = find_after_model(resume)
    else:
        if FLAGS.name is None:
            raise ValueError("--name is required when starting a new AFTER training")
        model_dir = os.path.join(FLAGS.out_path, FLAGS.name)
        restart_step = None
    os.makedirs(model_dir, exist_ok=True)

    # AFTER is trained on top of RAVE only: other codecs (e.g. AFTER autoencoders) must use AFTER directly
    check_rave_model(FLAGS.model, FLAGS.ckpt_name)

    # [First step] initialize RAVE model. AFTER calls the encode / decode methods of the embedding
    # model directly, so RAVE runs / checkpoints are first exported to a (non-streaming) .ts.
    logging.info("Initializing model %s"%FLAGS.model)
    if not os.path.exists(FLAGS.model):
        raise FileNotFoundError(FLAGS.model)
    if os.path.splitext(FLAGS.model)[1] == ".ts":
        rave_path = FLAGS.model
        embedding_name = FLAGS.embedding_name or os.path.splitext(os.path.basename(FLAGS.model))[0]
        if FLAGS.latent_size is not None or flags.FLAGS['fidelity'].present:
            logging.warning("--fidelity / --latent_size are ignored for .ts models: the exported latent size is used")
    else:
        # run folder, version folder or .ckpt (see docs/reference/runs.md)
        embedding_name = FLAGS.embedding_name or rave.get_run_name(rave.core.search_for_run(FLAGS.model, name=FLAGS.ckpt_name) or FLAGS.model) \
                         or Path(FLAGS.model).resolve().stem
        rave_dir = os.path.join(model_dir, "rave")
        existing = sorted(Path(rave_dir).glob(f"{embedding_name}*.ts")) if os.path.isdir(rave_dir) else []
        if resume is not None and len(existing) > 0 and not FLAGS.force:
            # keep the embedding model used at the beginning of the training
            rave_path = str(existing[0])
            logging.info(f"Reusing exported RAVE model {rave_path}")
        else:
            logging.info(f"Exporting RAVE run {FLAGS.model} to {rave_dir}...")
            rave_path = rave.scripting.export(FLAGS.model, out_path=rave_dir, name=embedding_name, streaming=False,
                                              fidelity=FLAGS.fidelity, latent_size=FLAGS.latent_size,
                                              ckpt_name=FLAGS.ckpt_name, ema_weights=FLAGS.ema_weights)
    rave_model = torch.jit.load(rave_path, map_location="cpu")
    rave_model.eval()
    rave_sr = rave_model.sr
    z_downsample = int(rave_model.encode_params[3])
    z_shape = int(rave_model.encode_params[2])
    n_channels = rave_model.n_channels
    encode_method = "encode"
    method_kwargs = {}
    logging.info(f"Model found at {rave_path}, dim={z_shape}, ratio={z_downsample}")
    ad.set_gin_constant('CHANNELS', n_channels)
    ad.set_gin_constant('SAMPLE_RATE', rave_sr)

    #
    # [Second step] init augmentation transforms
    logging.info("Initializing augmentation transforms...")
    rave_transforms = []
    if FLAGS.derivative: 
        rave_transforms += [adt.Derivator(sr=rave_model.sr, p=1.)]
    if FLAGS.normalize:
        rave_transforms += [adt.Normalize(p=1.)]
    augment_transforms = []
    if (FLAGS.stretch or FLAGS.shift):
        for i in range(FLAGS.n_augmentations): 
            pitch_transform = [adt.PitchShift(min_semitones=float(FLAGS.shift_range[0]), max_semitones=float(FLAGS.shift_range[1]), p=1.0, sr=rave_sr),
                                                 adt.TimeMask(min_band_part=0.07, max_band_part=0.15, fade=True, p=1.0, sr=rave_sr)]
            stretch_transform = [adt.TimeStretch(min_rate=float(FLAGS.stretch_range[0]), max_rate=float(FLAGS.stretch_range[1]), sr=rave_sr),
                                                   adt.TimeMask(min_band_part=0.07, max_band_part=0.15, fade=True, p=1.0, sr=rave_sr)]
            if (FLAGS.stretch and FLAGS.shift): 
                augment_transforms.append([*rave_transforms, *stretch_transform, *pitch_transform])
            elif (FLAGS.stretch):
                augment_transforms.append([*rave_transforms, *stretch_transform])
            elif (FLAGS.shift):
                augment_transforms.append([*rave_transforms, *pitch_transform])
    embedding_feature = ModuleEmbedding(module=rave_model, module_sr=rave_model.sr, 
                                        method=encode_method, method_kwargs=method_kwargs, 
                                        name=embedding_name, transforms=augment_transforms, device=device) 
    logging.info("Resulting embedding transform : %s..."%embedding_feature)

    
    # [Third step] init structure information and corresponding features if needed
    features = []
    logging.info(f"Parsing features for structure {FLAGS.structure}...")
    time_transform = None

    if (FLAGS.midi or FLAGS.structure == "midi"):
        midi_transform = adf.AfterMIDI(device=device)
        features.append(midi_transform)
    if (FLAGS.beat or FLAGS.structure == "beat"):
        time_transform = adf.BeatTrack(downsample=z_downsample, device=device)
        features.append(time_transform)
    logging.info(f"Parsed features : {features}")

    
    # [Fourth step] Update database
    logging.info("Parsing and updating databases...")
    chunk_size = None
    for data_path in FLAGS.db_path:
        dataset_metadata = ad.get_metadata_from_path(data_path)
        dataset_sr = dataset_metadata['sr']
        dataset_chunk_size = int(dataset_metadata.get('chunk_length') * dataset_metadata.get('sr'))
        chunk_size = dataset_chunk_size if chunk_size is None else min(dataset_chunk_size, chunk_size)

        # check latent embedding
        force_recompute_aug = FLAGS.force or FLAGS.force_augmentation
        if (embedding_name in ad.get_feature_names_from_path(data_path)) and not force_recompute_aug:
            # check if existing transform matches the current training
            saved_embedding = ad.get_features_from_path(data_path)[embedding_name]
            assert saved_embedding.n_augmentations == FLAGS.n_augmentations, \
                f"embedding {embedding_name} exists in data path {data_path}, but has {saved_embedding.n_augmentations} " \
                f"augmentations (asked: {FLAGS.n_augmentations}).\n" \
                "you can force recomputation of the augmentations with the --force_augmentation flag"
            for i, t in enumerate(saved_embedding.transforms): 
                if type(t) != type(augment_transforms[i]): 
                    logging.warning(f"transform #{i} ({augment_transforms[i]}) does not match with loaded embedding ({t})")

        # update embedding
        logging.info(f"Parsing embeddings for {data_path}...")
        with rave.core.GinEnv():
            ad.update_dataset(data_path, features = [embedding_feature], overwrite=force_recompute_aug, max_db_size=FLAGS.db_max_size)

        # update features
        force_recompute_feat = FLAGS.force or FLAGS.force_features
        if len(features) > 0:
            logging.info(f"Parsing features for {data_path}...")
            with rave.core.GinEnv():
                ad.update_dataset(data_path, features = features, overwrite=force_recompute_feat)


    # [Fifth step] Init datasets
    logging.info(f"Initializing datasets...")
    aug_keys = [f"aug_{i}" for i in range(FLAGS.n_augmentations)]
    data_keys = [f"{embedding_name}[0]->z", 
                 *[f"{embedding_name}[{i+1}]->{aug_keys[i]}" for i in range(FLAGS.n_augmentations)]]
    if FLAGS.structure == "beat": 
        data_keys += ["waveform"]
    elif FLAGS.structure == "midi": 
        data_keys += [f"{midi_transform.feature_name}->midi"]

    output_format = "{" + ",".join(data_keys) + "}"
    with gin.unlock_config():
        gin.bind_parameter("diffusion.utils.collate_fn.timbre_augmentation_keys", aug_keys)
                              
    dataset = []
    for d in FLAGS.db_path: 
        dataset.append(ad.datasets.AudioDataset(d, output_pattern=output_format, required_fields=['waveform', embedding_name]))
    if len(dataset) == 1: 
        dataset = dataset[0]
    else:
        dataset = ad.datasets.CombinedAudioDataset(dataset, FLAGS.db_weights, max_samples=FLAGS.db_max_size, use_cache=FLAGS.use_cache_for_multi)
    logging.info(f"Parsed dataset total length : {len(dataset)}...")
    partitions = dataset.split({'train': 0.98, 'val': 0.02}, load_if_available=FLAGS.db_partition, use_meta_if_available=True)
    train_dataset = partitions['train']; val_dataset = partitions['val']
    logging.info(f"Train size : {len(train_dataset)}; validation size : {len(val_dataset)}")

    # check data loading
    try: 
        data = train_dataset[0]
        data = val_dataset[0]
    except Exception as e: 
        logging.error("Caught error when testing data fetching. Got : ")
        raise e
   
    
    # [Sixth step] Init AFTER models
    logging.info(f"Initializing AFTER model")

    structure_type = "midi" if FLAGS.structure == "midi" else "audio"
    if not FLAGS.n_signal:
        FLAGS.n_signal = (chunk_size // 2)
    n_after_latents = FLAGS.n_signal // z_downsample
    logging.info("learned latent steps : %s"%(n_after_latents))


    if resume is not None:
        config_path = os.path.join(model_dir, "config.gin")
        logging.info('Taking checkpoint at : %s'%os.path.join(model_dir, after_ckpt_name))
        gin.parse_config_files_and_bindings([config_path], FLAGS.override)
        logging.info('Loaded config : %s'%gin.config_str())
    else:
        gin.parse_config_files_and_bindings(
            [add_gin_extension(FLAGS.config)],
            FLAGS.override + [f'IN_SIZE={z_shape}', f'STRUCTURE_TYPE="{structure_type}\"', f"N_SIGNAL={n_after_latents}",
                              f"SR={rave_sr}"],
        )

    with gin.unlock_config():
        gin.bind_parameter("diffusion.utils.collate_fn.ae_ratio", z_downsample)
        gin.bind_parameter("after.diffusion.model.fit.adversarial_weight", FLAGS.adv_weight)
        gin.bind_parameter('after.diffusion.utils.collate_fn.n_signal', n_after_latents)
        gin.bind_parameter('after.diffusion.utils.collate_fn.structure_type', structure_type)
        if FLAGS.val_every is not None:
            gin.bind_parameter("after.diffusion.model.fit.steps_valid", FLAGS.val_every)
        if FLAGS.save_every is not None:
            gin.bind_parameter("after.diffusion.model.fit.steps_save", FLAGS.save_every)

    rave_model = rave_model.to('cpu')

    after_transform = None
    if FLAGS.structure == "beat":
        assert time_transform
        after_transform = time_transform
    if FLAGS.arch == "rectified":
        after_model = RectifiedFlow(device=device, emb_model=rave_model, time_transform=after_transform)
    elif FLAGS.arch == "edm":
        after_model = EDM(device=device, emb_model=rave_model, time_transform=after_transform)

    # [Seventh step] Init data loaders
    logging.info(f"Initializing training")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=FLAGS.batch, 
        sampler = train_dataset.get_sampler(), 
        num_workers=FLAGS.workers,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=FLAGS.batch, 
        shuffle=False,
        num_workers=FLAGS.workers,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # [Eight step] Prepare checkpoints & logging number of paramters
    logging.info(f"Preparing training...")

    os.makedirs(model_dir, exist_ok=True)
    num_el = 0
    for p in after_model.net.parameters():
        num_el += p.numel()
    print("Number of parameters - unet : ", num_el / 1e6, "M")

    if after_model.encoder is not None:
        num_el = sum([p.numel() for p in after_model.encoder.parameters()])
        print("Number of parameters - encoder : ", num_el / 1e6, "M")

    if after_model.encoder_time is not None:
        num_el = sum([p.numel() for p in after_model.encoder_time.parameters()])
        print("Number of parameters - encoder_time : ", num_el / 1e6, "M")

    if after_model.classifier is not None:
        num_el = sum([p.numel() for p in after_model.classifier.parameters()])
        print("Number of parameters - classifier : ", num_el / 1e6, "M")

    if time_transform is not None: 
        print("With time transform : %s"%time_transform)

    ######### TRAINING #########
    d = {
        "model_dir": model_dir,
        "dataloader": train_loader,
        "validloader": val_loader,
        "restart_step": restart_step,
    }

    logging.info(f"Starting training...")
    if FLAGS.smoke_test: 
        import time
        os.makedirs("logs", exist_ok=True)
        with open(f"logs/log_model_{time.strftime('%y%m%d-%H%M%S')}.txt", 'w+') as f: print(after_model, file=f)
        with open(f"logs/log_config_{time.strftime('%y%m%d-%H%M%S')}.txt", 'w+') as f: print(gin.config_str(), file=f)
        with open(f"logs/log_opconfig_{time.strftime('%y%m%d-%H%M%S')}.txt", 'w+') as f: print(gin.operative_config_str(), file=f)
        
        logging.info('smoke_test succeeded!')
        pass
    else:
        after_model.fit(**d)
        logging.info(f"AFTER training done. Export it for nn~ with a *streaming* export of the same RAVE model "
                     f"(same latent size: {z_shape}), see the AFTER tutorial (docs/tutorials/06_prior.md):\n"
                     f"  rave export --run {FLAGS.model} --streaming --latent_size {z_shape}\n"
                     f"  python <AFTER>/after_scripts/export.py --model_path {model_dir} "
                     f"--emb_model_path <rave_streaming.ts> --step <step>")



if __name__ == "__main__": 
    parse_flags() 
    app.run(main)
