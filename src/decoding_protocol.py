"""Reference-free per-utterance budgets and EOS/cap accounting for ASR search."""
import torch


class DecodeTracker:
    def __init__(self, mask, max_len=None):
        if mask.ndim!=2 or mask.dtype!=torch.bool or not mask.size(0):
            raise ValueError('Expected a nonempty boolean encoder padding mask [B,T]')
        self.encoder_frames=(~mask).sum(1)
        if (self.encoder_frames<=0).any():
            raise ValueError('Every utterance must have at least one valid encoder frame')
        self.limits=self.encoder_frames+1  # At most one label per CTC frame, plus EOS.
        if max_len is not None:
            if isinstance(max_len,bool) or not isinstance(max_len,int) or max_len<=0:
                raise ValueError('An explicit decode guard must be a positive fixed integer')
            self.limits=self.limits.clamp_max(max_len)
        self.fixed_guard=max_len
        self.stop_steps=torch.zeros_like(self.limits)
        self.cap_hit=torch.zeros_like(self.limits,dtype=torch.bool)

    def inactive(self, finished, step):
        if finished.dtype!=torch.bool or finished.ndim not in (1,2) or finished.size(0)!=len(self.limits):
            raise ValueError('Finished flags must be boolean [B] or [B,K]')
        bounds=self.limits if finished.ndim==1 else self.limits[:,None]
        return finished | (step>=bounds)

    def update(self, finished, step):
        self.inactive(finished,step)  # Validate shapes before reducing beam flags.
        all_ended=finished if finished.ndim==1 else finished.all(-1)
        at_limit=step+1>=self.limits
        first_stop=(self.stop_steps==0)&(all_ended|at_limit)
        self.cap_hit|=first_stop&at_limit&~all_ended
        self.stop_steps=torch.where(first_stop,step+1,self.stop_steps)

    def finalize(self, sequences, finished, best=None, eos_id=2):
        """Trim inert carry tokens; never infer natural EOS from post-cap padding."""
        if (self.stop_steps==0).any():
            raise RuntimeError('Search did not observe termination or budget exhaustion for every row')
        if sequences.ndim!=4 or finished.ndim!=2 or sequences.shape[:2]!=finished.shape:
            raise ValueError('Expected sequences [B,K,L,C] and natural-EOS flags [B,K]')
        if sequences.size(0)!=len(self.limits) or sequences.size(2)<int(self.stop_steps.max()):
            raise ValueError('Output sequence shape is inconsistent with observed search steps')
        best=[0]*len(self.limits) if best is None else list(best)
        if len(best)!=len(self.limits):
            raise ValueError('Expected one selected beam per utterance')
        outputs,records=[],[]
        for b,index in enumerate(best):
            j=int(index)
            if not 0<=j<finished.size(1):raise ValueError('Selected beam is outside the beam array')
            sequence=sequences[b,j,:int(self.stop_steps[b])].detach().cpu().tolist()
            end=next((i for i,labels in enumerate(sequence) if eos_id in labels),None)
            natural=bool(finished[b,j])
            if natural!=(end is not None):
                raise ValueError('Natural-EOS flag disagrees with output before the row stopped')
            if natural:sequence=sequence[:end+1]
            outputs.append(sequence)
            records.append(dict(
                encoder_frames=int(self.encoder_frames[b]),effective_limit=int(self.limits[b]),
                limit_source='encoder_valid_frames_plus_eos',fixed_guard=self.fixed_guard,
                stop_steps=int(self.stop_steps[b]),selected_output_steps=len(sequence),
                termination='eos' if natural else 'cap',selected_eos=natural,
                search_cap_hit=bool(self.cap_hit[b]),completed_beams=int(finished[b].sum()),
                total_beams=finished.size(1)))
        return outputs,records
