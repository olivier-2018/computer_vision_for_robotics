function disp_img = getDisparity(...
    left_img, right_img, patch_radius, min_disp, max_disp)
% left_img and right_img are both H x W and you should return a H x W
% matrix containing the disparity d for each pixel of left_img. Set
% disp_img to 0 for pixels where the SSD and/or d is not defined, and for d
% estimates rejected in Part 2. patch_radius specifies the SSD patch and
% each valid d should satisfy min_disp <= d <= max_disp.

r = patch_radius;
patch_size = 2 * patch_radius + 1;

disp_img = zeros(size(left_img));

rows = size(left_img, 1);
cols = size(left_img, 2);

debug_ssds = false;  % Don't forget to replace parfor with for.
reject_outliers = true;
refine_estimate = true;

% parfor row = (1 + patch_radius):(rows-patch_radius)
for row = (1 + patch_radius):(rows-patch_radius)
    % In order to use parfor, the nested inner loop over the columns has
    % to work with a slice of the current column in the disparity image.
    % Once the inner loop is finised, the slice can be copied back to the
    % disparity image.    
    disp_slice = disp_img(row, :);
    for col = (1 + max_disp + patch_radius):(cols - patch_radius)
        % Here we construct what you can see in the left two subplots of
        % Fig. 4.
        left_patch = single(left_img((row-r):(row+r), (col-r):(col+r)));
        right_strip = single(right_img(...
            (row-r):(row+r), (col-r-max_disp):(col+r-min_disp)));
        
        % Transforming the patches into vectors so we can run them through
        % pdist2.
        lpvec = single(left_patch(:));
        rsvecs = single(zeros(patch_size^2, max_disp - min_disp + 1));
        for i = 1:patch_size
            rsvecs(((i-1)*patch_size+1):(i*patch_size), :) = ...
                right_strip(:, i:(max_disp - min_disp + i));
        end
        
        ssds = pdist2(lpvec', rsvecs', 'squaredeuclidean');
        
        if debug_ssds
            figure(10)
            subplot(1, 4, 1);
            imagesc(left_patch)
            axis equal
            axis off
            subplot(1, 4, 2:3);
            imagesc(right_strip)
            axis equal
            axis off
            subplot(1, 4, 4);
            plot(ssds, '-x');
            xlabel('d-d_{max}');
            ylabel('SSD(d)');
            pause
        end
        
        % The way the patches are set up, the argmin of ssds will not
        % directly be the disparity, but rather (max_disparity -
        % disparity). We call this "neg_disp".
        [min_ssd, neg_disp] = min(ssds);
        
        if reject_outliers
            if (nnz(ssds <= 1.5 * min_ssd) < 3 && neg_disp ~= 1 && ...
                    neg_disp ~= length(ssds))
                if ~refine_estimate
                    disp_slice(col) = max_disp - neg_disp;
                else
                    x = [neg_disp-1 neg_disp neg_disp+1];
                    p = polyfit(x, ssds(x), 2);
                    % Minimum of p(1)x^2 + p(2)x + p(3), converted from
                    % neg_disp to disparity as above.
                    disp_slice(col) = max_disp + p(2)/(2*p(1));
                end
            end
        else
            disp_slice(col) = max_disp - neg_disp;
        end
    end
    disp_img(row, :) = disp_slice;
end

end
