import csv

import matplotlib.pyplot as plt
import pandas as pd


def plot_rrt_star(rrt_star_file):
    df = pd.read_csv(rrt_star_file, names=['TYPE', 'OBSTACLE_SETUP', 'DURATION', 'RRT_FACTOR', 'SEED', 'NODES', 'PATH_NODES', 'PATH_LENGTH'])
    plot_tile = df.iloc[0, 0]
    print(plot_tile)
    df = df.drop('TYPE', axis=1)
    df = df[df['PATH_NODES'] != -1]
    df = df.drop('SEED', axis=1)

    unique_setups = df['OBSTACLE_SETUP'].unique()
    for setup in unique_setups:
        df_setup = df[df['OBSTACLE_SETUP'] == setup]
        df_setup = df_setup.drop('OBSTACLE_SETUP', axis=1)
        unique_rrt_factors = df_setup['RRT_FACTOR'].unique()
        fig, ax = plt.subplots()
        # fig.suptitle(f'{plot_tile} in Setup: {setup}')
        for rrt_factor in unique_rrt_factors:
            df_rrt_factor = df_setup[df_setup['RRT_FACTOR'] == rrt_factor]
            df_rrt_factor = df_rrt_factor.drop('RRT_FACTOR', axis=1)

            df_rrt_factor = df_rrt_factor.groupby(['DURATION']).mean()
            print(df_rrt_factor.head())
            df_rrt_factor.plot(y='PATH_LENGTH', ax=ax)
        ax.legend([f'β = {rrt_factor}' for rrt_factor in unique_rrt_factors])
        ax.set_ylabel('Path cost (m)')
        ax.set_xlabel('Duration (s)')
        plt.savefig(f'figures/RRT_Star_setup_{setup}.png', dpi=120, bbox_inches='tight')
        plt.show()


def plot_rrt_star_smart(rrt_star_smart_file):
    df = pd.read_csv(rrt_star_smart_file,
                     names=['TYPE', 'OBSTACLE_SETUP', 'DURATION', 'RRT_FACTOR', 'SMART_RATIO', 'SMART_RADIUS', 'SEED', 'NODES', 'PATH_NODES', 'PATH_LENGTH'])
    plot_tile = df.iloc[0, 0]
    print(plot_tile)
    df = df.drop('TYPE', axis=1)
    df = df[df['PATH_NODES'] != -1]
    df = df.drop('SEED', axis=1)

    unique_setups = df['OBSTACLE_SETUP'].unique()
    for setup in unique_setups:
        df_setup = df[df['OBSTACLE_SETUP'] == setup]
        df_setup = df_setup.drop('OBSTACLE_SETUP', axis=1)
        unique_smart_radius = df_setup['SMART_RADIUS'].unique()
        rrt_factor = df_setup['RRT_FACTOR'].unique()[0]
        df_setup = df_setup.drop('RRT_FACTOR', axis=1)
        n_radius = len(df_setup['SMART_RADIUS'].unique())

        fig, ax = plt.subplots(1, n_radius, sharey=True, figsize=(5*n_radius, 5))
        # fig.suptitle(f'{plot_tile} in Setup: {setup} with β = {rrt_factor}')
        for i, smart_radius in enumerate(unique_smart_radius):
            df_radius = df_setup[df_setup['SMART_RADIUS'] == smart_radius]
            df_radius = df_radius.drop('SMART_RADIUS', axis=1)
            unique_smart_ratio = df_radius['SMART_RATIO'].unique()
            for smart_ratio in unique_smart_ratio:
                df_ratio = df_radius[df_radius['SMART_RATIO'] == smart_ratio]
                df_ratio = df_ratio.drop('SMART_RATIO', axis=1)

                df_ratio = df_ratio.groupby(['DURATION']).mean()
                print(df_ratio.head())
                df_ratio.plot(y='PATH_LENGTH', title=f'Smart radius {smart_radius}', ax=ax[i])
            ax[i].legend([f's_ratio = {smart_ratio}' for smart_ratio in unique_smart_ratio])
            ax[i].set_ylabel('Path cost (m)')
            ax[i].set_xlabel('Duration (s)')
            print(ax)
        plt.savefig(f'figures/RRT_Star_Smart_setup_{setup}_rrtfactor_{rrt_factor}.png', dpi=120, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    rrt_star_final = 'RRT-Star.csv'
    plot_rrt_star(rrt_star_final)

    rrt_star_smart_1 = 'RRT-Star-Smart.csv'
    plot_rrt_star_smart(rrt_star_smart_1)
