import { test } from '@playwright/test';

const BASE_URL = 'http://127.0.0.1:5000/';

async function openDatasetMenuRoot(page) {
  await page.locator('#menu-dataset-view-main').getByText('example').hover();
  await page.waitForTimeout(100);
}

async function getCurrentMenuItems(page) {
  const menus = page.locator('.submenu[style*="display: block"], .submenu:visible');
  const menu = menus.last();
  return menu.locator('.dropdownmenuitem:visible');
}

async function replayMenuPath(page, pathIndexes) {
  await openDatasetMenuRoot(page);

  for (const index of pathIndexes) {
    const items = await getCurrentMenuItems(page);
    await items.nth(index).hover();
    await page.waitForTimeout(100);
  }
}

async function collectLeafMenuPaths(page, maxDepth = 10) {
  const leafPaths = [];

  async function walk(pathIndexes, pathLabels, depth) {
    if (depth > maxDepth) {
      throw new Error(`Too deep menu nesting: ${pathLabels.join(' > ')}`);
    }

    await replayMenuPath(page, pathIndexes);

    const items = await getCurrentMenuItems(page);
    const count = await items.count();

    for (let i = 0; i < count; i++) {
      await replayMenuPath(page, pathIndexes);

      const freshItems = await getCurrentMenuItems(page);
      const item = freshItems.nth(i);

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
  await replayMenuPath(page, pathIndexes.slice(0, -1));

  const items = await getCurrentMenuItems(page);
  await items.nth(pathIndexes[pathIndexes.length - 1]).click();
}

async function clickAcceptAndWaitDataset(page) {
  await Promise.all([
    page.waitForResponse(response =>
      response.url().includes('/dataset') && response.ok()
    ),
    page.getByRole('button', { name: 'Accept' }).click(),
  ]);
}

async function collectRadios(section) {
  const radios = section.locator('input[type="radio"]');
  const count = await radios.count();

  const result = [];

  for (let i = 0; i < count; i++) {
    const radio = radios.nth(i);

    result.push({
      index: i,
      value: await radio.getAttribute('value'),
      id: await radio.getAttribute('id'),
    });
  }

  return result;
}

async function collectCheckboxes(section) {
  const checkboxes = section.locator('input[type="checkbox"]');
  const count = await checkboxes.count();

  const result = [];

  for (let i = 0; i < count; i++) {
    const checkbox = checkboxes.nth(i);

    result.push({
      index: i,
      id: await checkbox.getAttribute('id'),
    });
  }

  return result;
}

async function setCheckboxById(page, checkboxId, checked) {
  const checkbox = page.locator(`input[type="checkbox"][id="${checkboxId}"]`);

  const current = await checkbox.isChecked();

  if (current === checked) {
    return;
  }

  // Не используем setChecked(), потому что input может быть скрыт,
  // а пользователь кликает по label / стилизованному элементу.
  await checkbox.evaluate((el, checked) => {
    el.checked = checked;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, checked);
}

async function checkRadioByIndex(section, radioIndex) {
  const radio = section.locator('input[type="radio"]').nth(radioIndex);

  await radio.evaluate(el => {
    el.checked = true;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  });
}

async function runAllVariableCombinations(page, datasetLabels) {
  const section = page.locator('#menu-dataset-var-view-main');

  const radios = await collectRadios(section);
  const checkboxes = await collectCheckboxes(section);

  const checkboxCount = checkboxes.length;
  const combinationsCount = 2 ** checkboxCount;

  let currentRadio = null;
  let currentMask = null;
  let currentCheckbox = null;

  try {
    for (let r = 0; r < radios.length; r++) {
      currentRadio = radios[r];

      for (let mask = 0; mask < combinationsCount; mask++) {
        currentMask = mask;

        const checkboxStates = {};

        for (let c = 0; c < checkboxCount; c++) {
          const checkbox = checkboxes[c];
          const checked = Boolean(mask & (1 << c));

          currentCheckbox = checkbox;
          checkboxStates[checkbox.id] = checked;

          await setCheckboxById(page, checkbox.id, checked);
        }

        currentCheckbox = null;

        await checkRadioByIndex(section, r);

        console.log('TRY COMBINATION:', {
          dataset: datasetLabels.join(' > '),
          radioIndex: r,
          radioValue: currentRadio.value,
          checkboxMask: mask,
          checkboxMaskBinary: mask.toString(2).padStart(checkboxCount, '0'),
          checkboxStates,
        });

        await clickAcceptAndWaitDataset(page);

        const editButton = page.locator('#menu-dataset-var-view-accept');

        if (await editButton.isVisible()) {
          await editButton.click();
        }
      }
    }
  } catch (error) {
    console.error('FAILED COMBINATION:', {
      dataset: datasetLabels.join(' > '),
      radio: currentRadio,
      checkboxMask: currentMask,
      checkboxMaskBinary: currentMask !== null
        ? currentMask.toString(2).padStart(checkboxCount, '0')
        : null,
      checkbox: currentCheckbox,
    });

    console.error(error);

    await page.pause();

    throw error;
  }
}

test('test all datasets and variable combinations', async ({ page }) => {
  await page.goto(BASE_URL);

  const leafPaths = await collectLeafMenuPaths(page);

  for (const path of leafPaths) {
    console.log('SELECT DATASET:', path.labels.join(' > '));

    await clickLeafMenuPath(page, path.indexes);

    await clickAcceptAndWaitDataset(page);

    await runAllVariableCombinations(page, path.labels);

    const datasetEditButton = page.locator('#menu-dataset-view-accept');

    if (await datasetEditButton.isVisible()) {
      await datasetEditButton.click();
    }
  }
});