
"""2"""
ax1 = plt.subplot(121); imshow(img, new_fig=False, title="")
plt.subplot(122, sharex=ax1, sharey=ax1); imshow(Fd, new_fig=False, title="")
plt.show(block=False)


"""4"""

ax1 = plt.subplot(221); imshow(img1, new_fig=False, title="")
plt.subplot(222, sharex=ax1, sharey=ax1); imshow(img2, new_fig=False, title="")
plt.subplot(223, sharex=ax1, sharey=ax1); imshow(img3, new_fig=False, title="")
plt.subplot(224, sharex=ax1, sharey=ax1); imshow(img4, new_fig=False, title="")
plt.show(block=False)