import { test, expect } from '@playwright/test';

  async function setCheckboxState(checkbox, checked) {
    const current = await checkbox.isChecked();

    if (current !== checked) {
      await checkbox.setChecked(checked, { force: true });
    }
  }

  async function clickAcceptAndWaitDataset(page) {
    await Promise.all([
      page.waitForResponse(response =>
        response.url().includes('/dataset') && response.ok()
      ),
      page.getByRole('button', { name: 'Accept' }).click(),
    ]);
  }

async function openDatasetMenuRoot(page) {
  await page.locator('#menu-dataset-view-main').getByText('example').hover();
}

async function getCurrentMenuItems(page) {
  const menus = page.locator('.dropdown-menu:visible, .submenu:visible, .menu:visible');
  const menu = menus.last();

  return menu.locator('> .dropdownmenuitem:visible');
}

async function collectLeafMenuPaths(page, maxDepth = 10) {
  const leafPaths = [];

  async function replayPath(pathIndexes) {
    await openDatasetMenuRoot(page);

    for (const index of pathIndexes) {
      const items = await getCurrentMenuItems(page);
      await items.nth(index).hover();
      await page.waitForTimeout(100);
    }
  }

  async function walk(pathIndexes, pathLabels, depth) {
    if (depth > maxDepth) {
      throw new Error(`Too deep menu nesting: ${pathLabels.join(' > ')}`);
    }

    await replayPath(pathIndexes);

    const items = await getCurrentMenuItems(page);
    const count = await items.count();

    for (let i = 0; i < count; i++) {
      await replayPath(pathIndexes);

      const freshItems = await getCurrentMenuItems(page);
      const item = freshItems.nth(i);

      if (!(await item.isVisible())) {
        continue;
      }

      const label = (await item.innerText()).trim();
      const hasSubmenu = await item.evaluate(el =>
        el.classList.contains('has-submenu')
      );

      if (hasSubmenu) {
        await item.hover();
        await page.waitForTimeout(100);

        await walk([...pathIndexes, i], [...pathLabels, label], depth + 1);
      } else {
        leafPaths.push({
          indexes: [...pathIndexes, i],
          labels: [...pathLabels, label],
        });
      }
    }
  }

  await walk([], [], 0);

  return leafPaths;
}

async function clickLeafMenuPath(page, pathIndexes) {
  await openDatasetMenuRoot(page);

  for (let i = 0; i < pathIndexes.length; i++) {
    const items = await getCurrentMenuItems(page);
    const item = items.nth(pathIndexes[i]);

    if (i === pathIndexes.length - 1) {
      await item.click();
    } else {
      await item.hover();
      await page.waitForTimeout(100);
    }
  }
}

test('test datasets 1', async ({ page }) => {
  await page.goto('http://127.0.0.1:5000/');
  // await page.goto('http://10.10.53.186:8090/');

  // await page.locator('#menu-dataset-view-main').getByText('example').hover();
  // await page.getByText('custom').hover();
  // await page.getByTitle('Graphs 1\nNodes: [34]\nDirected: False\nHetero: False').click();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');

  const leafPaths = await collectLeafMenuPaths(page);

  for (const path of leafPaths) {
    console.log('Select dataset:', path.labels.join(' > '));

    await clickLeafMenuPath(page, path.indexes);

    await page.getByRole('button', { name: 'Accept' }).click();

    await page.waitForResponse('**/dataset');

    // start features-labels cycle
    const section = page.locator('#menu-dataset-var-view-main');
    const radios = section.locator('input[type="radio"]');
    const radioCount = await radios.count();

    let currentRadioIndex = null;
    let currentCheckboxMask = null;
    let currentCheckboxIndex = null;

    try {
      for (let r = 0; r < radioCount; r++) {
        currentRadioIndex = r;

        const radio = radios.nth(r);

        const checkboxes = section.locator('input[type="checkbox"]:visible');
        const checkboxCount = await checkboxes.count();
        const combinationsCount = 2 ** checkboxCount;

        for (let mask = 0; mask < combinationsCount; mask++) {
          currentCheckboxMask = mask;

          console.log('TRY COMBINATION:', {
            radioIndex: r,
            checkboxMask: mask,
            checkboxMaskBinary: mask.toString(2).padStart(checkboxCount, '0'),
          });

          // set checkbox combination
          for (let c = 0; c < checkboxCount; c++) {
            currentCheckboxIndex = c;

            const shouldBeChecked = Boolean(mask & (1 << c));
            await setCheckboxState(checkboxes.nth(c), shouldBeChecked);
          }

          currentCheckboxIndex = null;

          // set radio
          await radio.check({ force: true });

          // press Accept and wait response
          await clickAcceptAndWaitDataset(page);

          // Edit dataset-var block
          await page.locator('#menu-dataset-var-view-accept').click();
        }
      }
    } catch (error) {
      console.error('FAILED COMBINATION:', {
        radioIndex: currentRadioIndex,
        checkboxMask: currentCheckboxMask,
        checkboxMaskBinary: currentCheckboxMask !== null
          ? currentCheckboxMask.toString(2).padStart(checkboxCount, '0')
          : null,
        checkboxIndex: currentCheckboxIndex,
      });

      console.error(error);

      await page.pause();

      throw error;
    }
    // Edit dataset raw block
    await page.locator('#menu-dataset-view-accept').click();
  }



  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: '3comms (3 classes)' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: '4comms (4 classes)' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: 'default' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('checkbox', { name: '-hot over nodes (size=34)' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: 'and 33 nodes (2 classes)' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: '2comms (2 classes)' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: '3comms (3 classes)' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: '4comms (4 classes)' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('div').filter({ hasText: /^Edit$/ }).nth(3).click();
  // await page.getByRole('checkbox', { name: 'ones (size=10)' }).uncheck();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: 'and 33 nodes (2 classes)' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.locator('div').filter({ hasText: /^2comms \(2 classes\)$/ }).click();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: 'default' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-view-accept').click();
  //
  // await page.locator('#menu-dataset-view-main').getByText('example').hover();
  // await page.getByTitle('Graphs 1\nNodes: [8]\nDirected: False\nHetero: False').click();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.getByRole('checkbox', { name: 'b (size=3)' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: 'threeClasses (3 classes)' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.locator('#menu-dataset-var-view-labelings-node-regressionregression').check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.locator('#menu-dataset-var-view-labelings-edge-regressionregression').check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: 'default' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.locator('#menu-dataset-var-view-node-1hot-input').check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: 'binary (2 classes)' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.locator('div').filter({ hasText: /^threeClasses \(3 classes\)$/ }).click();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.locator('#menu-dataset-var-view-labelings-node-regressionregression').check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.locator('#menu-dataset-var-view-labelings-edge-regressionregression').check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: 'default' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('checkbox', { name: 'b (size=3)' }).uncheck();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-view-accept').click();
  //
  // await page.locator('#menu-dataset-view-main').getByText('example').hover();
  // await page.getByTitle('Graphs 1\nNodes: [8]\nDirected: True\nHetero: False').click();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.getByRole('checkbox', { name: 'b (size=1)' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: 'default' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('checkbox', { name: '-hot over nodes (size=8)' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('checkbox', { name: 'a (size=1)' }).uncheck();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: 'binary (2 classes)' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('checkbox', { name: 'b (size=1)' }).uncheck();
  // await page.getByRole('checkbox', { name: 'a (size=1)' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: 'default' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-view-accept').click();
  //
  // await page.locator('#menu-dataset-view-main').getByText('example').hover();
  // await page.getByTitle('Graphs 3\nNodes: [3, 4, 5]\nDirected: False\nHetero: False').click();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.getByText('threeClasses (3 classes)').click();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: 'regression' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByText('default').click();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-view-accept').click();
  //
  // await page.locator('#menu-dataset-view-main').getByText('example').hover();
  // await page.getByTitle('Graphs 8\nNodes: [5, 4, 4, 8, 6, 7, 7, 9]\nDirected: True\nHetero: False').click();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.getByRole('checkbox', { name: 'b (size=2)' }).check();
  // await page.getByRole('checkbox', { name: 'a (size=2)' }).uncheck();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: 'default' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('checkbox', { name: 'a (size=2)' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('radio', { name: 'binary (2 classes)' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');
  // await page.locator('#menu-dataset-var-view-accept').click();
  // await page.getByRole('checkbox', { name: 'b (size=2)' }).uncheck();
  // await page.getByRole('radio', { name: 'default' }).check();
  // await page.getByRole('button', { name: 'Accept' }).click();
  // await page.waitForResponse('**/dataset');

});
