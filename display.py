import matplotlib.pyplot as plt


# Displaying the data
def displayData(displaySize, data, selected, title):
    # setting up our plot
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(title, fontsize=32)
    # configuring the number of images to display
    columns = displaySize
    rows = displaySize

    for i in range(columns * rows):
        # if we want to display multiple images,
        # then 'selected' is a vector. Check if it is here:
        if hasattr(selected, "__len__"):
            img = data[selected[i]]
        else:
            img = data[selected]
        img = img.reshape(20, 20).transpose()
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img)

        # We could also use plt.show(), but repl
        # can't  display it. So let's insted save it
        # into a file
    plt.savefig('plots/' + title)

    return None