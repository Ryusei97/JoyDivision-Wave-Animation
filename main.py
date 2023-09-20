from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from tqdm import tqdm
from math import exp, sin, pi
from moviepy.editor import *
import random
import os


def R(_n):
    random.seed(_n)
    return random.random()

class AudioVis:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.samplerate, self.data = wavfile.read(file_path)
        self.length = self.data.shape[0] / self.samplerate
        self.left_channel = self.data[:, 0]
        self.right_channel = self.data[:, 1]
        self.time = np.linspace(0., self.length, self.data.shape[0])
        self.fps = 10
        self.right = []
        self.left = []
        self.base_wave = []
        self.processed = []
        
        self.generate_base()
        self.base_l = len(self.base_wave[0][0])

    def plot_raw(self):
        '''
        Plots the raw audio file
        '''
        time = self.time
        plt.plot(time, self.left_channel, label="Left channel")
        plt.plot(time, self.right_channel, label="Right channel")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()

    def plot_spectrogram(self):
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Spectrograms')
        ax1.specgram(self.left_channel, Fs=self.samplerate, cmap="magma", vmin=-40, vmax=60)
        ax1.set_title('Left Channel')
        ax1.get_xaxis().set_visible(False)
        ax1.set_ylabel('Frequency (Hz)')
        ax2.specgram(self.right_channel, Fs=self.samplerate, cmap="magma", vmin=-40, vmax=60)
        ax2.set_title('Right Channel')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Frequency (Hz)')

        plt.show()

    def separate_freq(self):
        '''
        Separate the wav file into reformatted array
        '''
        def rolling_ave(a, n=10):
            a = np.pad(a, (2*n, 2*n), constant_values=(0,0))
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n


        if self.processed:
            return

        seg = 44100 // self.fps
        back = 2*seg-len(self.right_channel)%seg if len(self.right_channel)%seg != 0 else seg-len(self.right_channel)%seg

        r = np.pad(self.right_channel, (2*seg, back), constant_values=(0, 0))
        r = r.reshape((len(r)//seg, seg))
        l = np.pad(self.left_channel, (2*seg, back), constant_values=(0, 0))
        l = l.reshape((len(l)//seg, seg))

        for i in range(2, len(l)):
            self.right.append(r[i-2:i+1].flatten())
            self.left.append(l[i-2:i+1].flatten())

        shifts = [random.randint(-20, 20) for i in range(80)]
        for i in range(len(self.right)):
            r_ = spectrogram(x=self.right[i], fs=self.samplerate, nperseg=370, mode='magnitude')[2].T
            l_ = spectrogram(x=self.left[i], fs=self.samplerate, nperseg=370, mode='magnitude')[2].T
            new_r = []
            for j, row in enumerate(r_):
                c = row[::2]
                d = row[1::2][::-1]
                e = np.concatenate((d,c))
                e = rolling_ave(e)
                f = np.pad(e, ((self.base_l-len(e))//2,0), constant_values=(0,0))
                f = np.pad(f, (0,self.base_l-len(f)), constant_values=(0,0))
                new_r.append(np.roll(f, shifts[j]))

            new_l = []
            for j, row in enumerate(l_):
                c = row[::2]
                d = row[1::2][::-1]
                e = np.concatenate((d,c))
                e = rolling_ave(e)
                f = np.pad(e, ((self.base_l-len(e))//2,0), constant_values=(0,0))
                f = np.pad(f, (0,self.base_l-len(f)), constant_values=(0,0))
                new_l.append(np.roll(f, shifts[40+j]))
            
            # g = np.concatenate((np.array(new_r), np.array(new_l)), axis=0)
            g = []
            for j in range(len(new_l)):
                g.append(new_r[j])
                g.append(new_l[j])
            g = np.array(g)
            self.processed.append(g)

        self.processed = 10*np.log10(np.array(self.processed))
        self.processed[self.processed < -20] = -20
        self.processed = 10*(self.processed-np.min(self.processed))/(np.max(self.processed)-np.min(self.processed))

        return
        
    def animate_spectrogram(self):
        '''
        Animates the separated fft data
        '''
        self.separate_freq()

        # segment_length = self.samplerate // 10
        # num_segments = len(self.right_channel) // segment_length

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), facecolor='white')
        fig.suptitle('Spectrograms')
        plt.style.use("ggplot")

        def animate(i):
            # start_idx = i * segment_length
            # end_idx = start_idx + segment_length
            # y1 = self.right_channel[start_idx:end_idx]
            # y2 = self.left_channel[start_idx:end_idx]

            y1 = self.right[i]
            y2 = self.left[i]

            ax1.clear()
            ax2.clear()
            ax1.specgram(y2, Fs=self.samplerate, cmap="magma", vmin=-40, vmax=60)
            ax1.set_title('Left Channel')
            ax1.get_xaxis().set_visible(False)
            ax1.set_ylabel('Frequency (Hz)')
            ax2.specgram(y1, Fs=self.samplerate, cmap="magma", vmin=-40, vmax=60)
            ax2.set_title('Right Channel')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Frequency (Hz)')

        anim = FuncAnimation(fig, animate, frames=len(self.right), blit=False, interval=0)

        # plt.show()
        anim.save('fft.gif', writer = 'pillow', fps = self.fps)

        return 


    def visualize(self):
        ''' 
        Creates a matplotlib visualization where each of the separated frequencies are 
        plotted on a single stacked plot. It should resemble Joy Division's album cover. 
        '''
        self.separate_freq()
        

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 8), facecolor='black')
 
        plt.style.use("ggplot")
        
        def animate(i):
            y = self.processed[i]
            axes.clear()
            axes.set_ylim(-5, 90)
            axes.set_xlim(-60, 160)
            axes.axis('off')

            zorder = 0
            for j, wave in enumerate(self.base_wave):
                temp1 = wave[i%18]
                temp2 = y[j]
                axes.plot(self.xs, wave[i%18] + y[j]*R(j), color='white', zorder=zorder)
                axes.fill_between(self.xs, wave[i%18]+y[j]*R(j), color='black', alpha=1.0, zorder=zorder)
                zorder += 1


        anim = FuncAnimation(fig, animate, frames=len(self.processed), blit=False, interval=0)

        anim.save('vis.gif', writer = 'pillow', fps = self.fps)

        return

    def generate_base(self):
        if os.path.exists('base_wave.npy'):
            self.base_wave = np.load('base_wave.npy')
            self.xs = np.linspace(-50, 150, 380)
            return

        # Change number of points to change smoothness of the graph
        self.xs = np.linspace(-50, 150, 380)

        for m in tqdm(range(1, 81, 1)):
            lines = []
            for t in np.linspace(0, (6.3*18)/19, 18):
                l = []
                for x in self.xs:
                    small = [4*sin(2*pi*R(4*m)+t+R(2*n*m)*2*pi)*exp(-(0.3*x+30-100*R(2*n*m))**2.0/20.0) for n in range(1, 31, 1)]
                    d = 80 - m + 0.2*sin(2*pi*R(6*m) + sum(small))
                    l.append(d)
                lines.append(l)
            self.base_wave.append(lines)
        
        np.save('base_wave.npy', self.base_wave)

        return
    
    def combine(self):
        gif_clip = VideoFileClip('vis.gif')

        audio_clip = AudioFileClip(self.file_path)

        # Set the audio of the GIF to the audio from the WAV file
        final_clip = gif_clip.set_audio(audio_clip)

        final_clip.write_videofile('combined_video.mp4', codec='libx264', audio_codec='aac')
        # final_clip.preview(fps=30)
    

# Music from https://pixabay.com/
if __name__ == '__main__':
    audio = AudioVis(file_path='risk.wav')
    # audio.plot_raw()
    audio.visualize()
    audio.combine()

    
