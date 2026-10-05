import { create } from 'zustand';
import type { AxiosResponse } from 'axios';
import {
  getServiceStatus,
  getTrainProgress,
  type TrainProgress,
  type ServiceStatusRes
} from '@/service/train';
import type { CommonResponse } from '@/types/responseModal';

export type ModelStatus = 'seed_identity' | 'memory_upload' | 'training' | 'trained';

export enum Status {
  SEED_IDENTITY = 'seed_identity',
  MEMORY_UPLOAD = 'memory_upload',
  TRAINING = 'training',
  TRAINED = 'trained'
}

export const statusRankMap = {
  [Status.SEED_IDENTITY]: 0,
  [Status.MEMORY_UPLOAD]: 1,
  [Status.TRAINING]: 2,
  [Status.TRAINED]: 2
};

interface ModelState {
  status: ModelStatus;
  error: boolean;
  isTraining: boolean;
  serviceStarted: boolean;
  isServiceStarting: boolean;
  isServiceStopping: boolean;
  trainingProgress: TrainProgress;
  trainSuspended: boolean;
  setStatus: (status: ModelStatus) => void;
  setError: (error: boolean) => void;
  setIsTraining: (isTraining: boolean) => void;
  fetchServiceStatus: () => Promise<AxiosResponse<CommonResponse<ServiceStatusRes>>>;
  setServiceStarting: (isStarting: boolean) => void;
  setServiceStopping: (isStopping: boolean) => void;
  setTrainingProgress: (progress: TrainProgress) => void;
  setTrainSuspended: (suspended: boolean) => void;
  checkTrainStatus: () => Promise<void>;
  resetTrainingState: () => void;
}

const defaultTrainingProgress: TrainProgress = {
  current_stage: 'downloading_the_base_model',
  overall_progress: 0,
  stages: [
    {
      current_step: null,
      name: 'Downloading the Base Model',
      progress: 0,
      status: 'pending',
      steps: [{ completed: false, name: 'Model Download', status: 'pending' }]
    },
    {
      current_step: null,
      name: 'Activating the Memory Matrix',
      progress: 0,
      status: 'pending',
      steps: [
        { completed: false, name: 'List Documents', status: 'pending' },
        { completed: false, name: 'Generate Document Embeddings', status: 'pending' },
        { completed: false, name: 'Process Chunks', status: 'pending' },
        { completed: false, name: 'Chunk Embedding', status: 'pending' }
      ]
    },
    {
      current_step: null,
      name: 'Synthesize Your Life Narrative',
      progress: 0.0,
      status: 'pending',
      steps: [
        { completed: false, name: 'Extract Dimensional Topics', status: 'pending' },
        { completed: false, name: 'Map Your Entity Network', status: 'pending' }
      ]
    },
    {
      current_step: null,
      name: 'Prepare Training Data for Deep Comprehension',
      progress: 0.0,
      status: 'pending',
      steps: [
        { completed: false, name: 'Decode Preference Patterns', status: 'pending' },
        { completed: false, name: 'Reinforce Identity', status: 'pending' },
        { completed: false, name: 'Augment Content Retention', status: 'pending' }
      ]
    },
    {
      current_step: null,
      name: 'Training to create Second Me',
      progress: 0.0,
      status: 'pending',
      steps: [
        { completed: false, name: 'Train', status: 'pending' },
        { completed: false, name: 'Merge Weights', status: 'pending' },
        { completed: false, name: 'Convert Model', status: 'pending' }
      ]
    }
  ],
  status: 'pending'
};

export const useTrainingStore = create<ModelState>((set, get) => ({
  status: 'seed_identity',
  isTraining: false,
  serviceStarted: false,
  isServiceStarting: false,
  isServiceStopping: false,
  error: false,
  trainingProgress: defaultTrainingProgress,
  trainSuspended: false,
  setStatus: (status) => {
    const preStatus = get().status;

    //Only trained and running can be interchanged.
    if (statusRankMap[status] < statusRankMap[preStatus]) {
      return;
    }

    set({ status });
  },
  fetchServiceStatus: () => {
    return getServiceStatus().then((res) => {
      if (res.data.code === 0) {
        const isRunning = res.data.data.is_running;

        if (isRunning) {
          set({ serviceStarted: true });
        } else {
          set({ serviceStarted: false });
        }
      }

      return res;
    });
  },
  setError: (error) => set({ error }),
  setIsTraining: (isTraining) => set({ isTraining }),
  setServiceStarting: (isStarting) => set({ isServiceStarting: isStarting }),
  setServiceStopping: (isStopping) => set({ isServiceStopping: isStopping }),
  setTrainingProgress: (progress) => set({ trainingProgress: progress }),
  setTrainSuspended: (suspended) => set({ trainSuspended: suspended }),
  resetTrainingState: () => set({ trainingProgress: defaultTrainingProgress }),
  checkTrainStatus: async () => {
    const config = JSON.parse(localStorage.getItem('trainingParams') || '{}');

    set({ error: false });

    try {
      const res = await getTrainProgress({
        model_name: config.model_name || 'Qwen3-1.7B'
      });

      if (res.data.code === 0) {
        const data = res.data.data;
        const { overall_progress, status } = data;

        const newProgress = data;

        if (newProgress.status === 'failed') {
          set({ error: true });
        }

        set((state) => {
          const newState = {
            ...state,
            trainingProgress: newProgress
          };

          if (status === 'suspended' || status === 'failed') {
            newState.trainSuspended = true;
          }

          // If total progress is 100%, set status to trained
          if (overall_progress === 100) {
            newState.status = 'trained';
          }
          // If there's any progress but not complete, set status to training
          else if (overall_progress > 0) {
            newState.status = 'training';
          }

          return newState;
        });
      }
    } catch (error) {
      console.error('Error checking training status:', error);
      set({ error: true });
    }
  }
}));
